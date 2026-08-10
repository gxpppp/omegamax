// Implementation of the C++ MCTS search engine (see mcts_engine.h).
//
// The search loop mirrors omigamax.mcts.mcts.run_search on the batched-
// evaluator path exactly (selection -> terminal / pending checks -> claim /
// submit / flush / expand / backup), so a deterministic evaluator yields the
// same tree, visit counts and policy as the Python reference.

#include "mcts_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace omigamax {

namespace {

// Game outcome from the mover's perspective at a terminal position -- mirrors
// mcts.terminal_value (color is always threaded in the engine, never None).
double terminal_value(const Board& b, double komi, int color) {
    const std::string winner = b.winner(komi);
    if (winner.empty()) {
        return 0.0;
    }
    const bool current_is_black = (color == BLACK);
    if ((winner == "B") == current_is_black) {
        return 1.0;
    }
    return -1.0;
}

}  // namespace

// ---------------------------------------------------------------------------
// construction / import
// ---------------------------------------------------------------------------

CppMCTSEngine::CppMCTSEngine(py::object py_root, py::object evaluator,
                             int simulations, double c_puct, double komi,
                             int virtual_loss, int batch_size,
                             py::object make_node, py::object noisy_prior)
    : py_root_(std::move(py_root)),
      evaluator_(std::move(evaluator)),
      make_node_(std::move(make_node)),
      simulations_(simulations),
      c_puct_(c_puct),
      komi_(komi),
      virtual_loss_(virtual_loss),
      batch_size_(batch_size) {
    // Copy of the Python root's move history (prefix of every shell's moves).
    root_moves_ = py::list(py_root_.attr("board").attr("moves"));

    if (!noisy_prior.is_none()) {
        py::dict d = py::cast<py::dict>(noisy_prior);
        for (const auto item : d) {
            noisy_[py::cast<int>(item.first)] =
                py::cast<double>(item.second);
        }
    }

    root_idx_ = import_node(py_root_, -1, -1);
}

int CppMCTSEngine::import_node(const py::object& pynode, int parent_idx,
                               int parent_color) {
    py::object pyboard = pynode.attr("board");
    const int size = py::cast<int>(pyboard.attr("size"));
    std::vector<std::int8_t> state =
        py::cast<std::vector<std::int8_t>>(pyboard.attr("_state"));
    const int move_count = static_cast<int>(py::len(pyboard.attr("moves")));
    const int pass_count = py::cast<int>(pyboard.attr("pass_count"));
    int last_captured = -1;
    py::object lcp = pyboard.attr("last_captured_point");
    if (!lcp.is_none()) {
        py::tuple t = py::cast<py::tuple>(lcp);
        last_captured = py::cast<int>(t[0]) * size + py::cast<int>(t[1]);
    }

    // Threaded color: mirrors Node.color (an explicit _color wins, otherwise
    // a non-root child flips the parent's mover; the root falls back to
    // move-count parity).
    int color = 0;
    py::object c = pynode.attr("_color");
    if (!c.is_none()) {
        color = py::cast<int>(c);
    } else if (parent_idx >= 0) {
        color = 3 - parent_color;
    } else {
        color = (move_count % 2 == 0) ? BLACK : WHITE;
    }

    const double prior = py::cast<double>(pynode.attr("prior"));
    const int visit_count = py::cast<int>(pynode.attr("visit_count"));
    const double value_sum = py::cast<double>(pynode.attr("value_sum"));
    const int virtual_loss = py::cast<int>(pynode.attr("virtual_loss"));

    const int idx = alloc_node();
    CppNode& n = nodes_[idx];
    n.parent = parent_idx;
    n.color = color;
    n.prior = prior;
    n.visit_count = visit_count;
    n.value_sum = value_sum;
    n.virtual_loss = virtual_loss;
    n.board_index = static_cast<int>(boards_.size());
    boards_.emplace_back(size);
    boards_[n.board_index].load_position(state, move_count, pass_count,
                                         last_captured);
    py::object lm = pynode.attr("legal_moves");
    if (!lm.is_none()) {
        n.legal_moves = py::cast<std::vector<int>>(lm);
        n.legal_moves_known = true;
    }

    py::dict children = py::cast<py::dict>(pynode.attr("children"));
    std::vector<std::pair<int, py::object>> items;
    items.reserve(children.size());
    for (const auto item : children) {
        items.emplace_back(py::cast<int>(item.first),
                           py::reinterpret_borrow<py::object>(item.second));
    }
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [action, child] : items) {
        const int ci = import_node(child, idx, color);
        nodes_[idx].children.push_back({action, ci});
    }
    nodes_[idx].expanded = !nodes_[idx].children.empty();
    return idx;
}

// ---------------------------------------------------------------------------
// pools
// ---------------------------------------------------------------------------

int CppMCTSEngine::alloc_node() {
    nodes_.emplace_back();
    return static_cast<int>(nodes_.size()) - 1;
}

int CppMCTSEngine::clone_board(int board_index) {
    boards_.push_back(boards_[board_index]);
    return static_cast<int>(boards_.size()) - 1;
}

int CppMCTSEngine::child_of(int node_idx, int action) const {
    for (const auto& [a, ci] : nodes_[node_idx].children) {
        if (a == action) {
            return ci;
        }
    }
    throw std::runtime_error("selection produced a non-child action");
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

int CppMCTSEngine::select_child(int node_idx) const {
    const CppNode& n = nodes_[node_idx];
    const bool at_root = (node_idx == root_idx_);
    const double sqrt_parent = std::sqrt(static_cast<double>(n.visit_count));
    int best_action = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const auto& [action, ci] : n.children) {
        const CppNode& child = nodes_[ci];
        double prior = child.prior;
        if (at_root) {
            const auto it = noisy_.find(action);
            if (it != noisy_.end()) {
                prior = it->second;
            }
        }
        const int eff = child.visit_count + child.virtual_loss;
        const double q = child.visit_count > 0
                             ? (child.value_sum /
                                static_cast<double>(child.visit_count))
                             : 0.0;
        const double ucb =
            q + c_puct_ * prior * sqrt_parent / (1.0 + static_cast<double>(eff));
        if (ucb > best_score) {
            best_score = ucb;
            best_action = action;
        }
    }
    return best_action;
}

void CppMCTSEngine::ensure_legal_moves(int node_idx) {
    if (nodes_[node_idx].legal_moves_known) {
        return;
    }
    nodes_[node_idx].legal_moves =
        boards_[nodes_[node_idx].board_index].legal_actions(
            nodes_[node_idx].color);
    nodes_[node_idx].legal_moves_known = true;
}

void CppMCTSEngine::expand(int node_idx, const float* prior) {
    // The legal-move list is copied first: alloc_node() can reallocate the
    // node pool, which would invalidate a range-for over the node's vectors.
    const std::vector<int> legal = nodes_[node_idx].legal_moves;
    const int size = boards_[nodes_[node_idx].board_index].size();
    const int color = nodes_[node_idx].color;
    const int pass = size * size;
    for (const int action : legal) {
        const int ci = alloc_node();
        nodes_[ci].parent = node_idx;
        nodes_[ci].board_index = clone_board(nodes_[node_idx].board_index);
        Board& cb = boards_[nodes_[ci].board_index];
        if (action == pass) {
            cb.pass_move(color);
        } else {
            cb.play(action / size, action % size, color, /*check_legal=*/false);
        }
        nodes_[ci].color = 3 - color;
        nodes_[ci].prior = static_cast<double>(prior[action]);
        nodes_[node_idx].children.push_back({action, ci});
    }
    nodes_[node_idx].expanded = true;
}

void CppMCTSEngine::backup(const std::vector<int>& path, double v) {
    for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
        nodes_[path[i]].visit_count += 1;
        nodes_[path[i]].value_sum += v;
        v = -v;
    }
}

bool CppMCTSEngine::is_pending(int node_idx) const {
    for (const auto& item : pending_) {
        if (item.node_idx == node_idx) {
            return true;
        }
    }
    return false;
}

void CppMCTSEngine::flush_batch() {
    if (pending_.empty()) {
        return;
    }
    try {
        py::list results =
            py::cast<py::list>(evaluator_.attr("flush")());
        if (py::len(results) != static_cast<py::ssize_t>(pending_.size())) {
            throw std::runtime_error(
                "batched evaluator returned a different number of results "
                "than submitted leaves");
        }
        for (size_t i = 0; i < pending_.size(); ++i) {
            PendingItem& item = pending_[i];
            py::tuple r = py::cast<py::tuple>(results[i]);
            py::object fnode = r[0];
            if (!fnode.is(item.shell)) {
                throw std::runtime_error(
                    "batched evaluator returned a leaf in a different order "
                    "than submitted");
            }
            py::array_t<float> prior = py::cast<py::array_t<float>>(r[1]);
            const double value = py::cast<double>(r[2]);
            expand(item.node_idx, prior.data());
            backup(item.path, value);
        }
    } catch (...) {
        // virtual-loss claims are released on any evaluator failure, mirroring
        // the try/finally in run_search's flush_batch.
        for (auto& item : pending_) {
            nodes_[item.node_idx].virtual_loss -= virtual_loss_;
        }
        pending_.clear();
        throw;
    }
    for (auto& item : pending_) {
        nodes_[item.node_idx].virtual_loss -= virtual_loss_;
    }
    pending_.clear();
}

void CppMCTSEngine::run() {
    int sims_done = 0;
    while (sims_done < simulations_) {
        // -- selection: descend until an unexpanded leaf --
        int node_idx = root_idx_;
        std::vector<int> path;
        path.push_back(node_idx);
        while (nodes_[node_idx].expanded) {
            const int action = select_child(node_idx);
            node_idx = child_of(node_idx, action);
            path.push_back(node_idx);
        }

        // -- terminal leaves never go through the evaluator --
        ensure_legal_moves(node_idx);
        if (board(node_idx).is_terminal()) {
            const double value =
                terminal_value(board(node_idx), komi_, nodes_[node_idx].color);
            backup(path, value);
            sims_done += 1;
            if (static_cast<int>(pending_.size()) >= batch_size_) {
                flush_batch();
            }
            continue;
        }

        // -- re-selecting an already-pending leaf: flush to free it --
        if (is_pending(node_idx)) {
            if (!pending_.empty()) {
                flush_batch();
            }
            continue;  // no simulation consumed; retry on the new tree
        }

        // -- expandable leaf: claim virtual loss, submit the shell --
        nodes_[node_idx].virtual_loss += virtual_loss_;
        py::object shell = get_shell(node_idx);
        evaluator_.attr("submit")(shell);
        pending_.push_back({node_idx, std::move(path), shell});
        sims_done += 1;
        if (static_cast<int>(pending_.size()) >= batch_size_) {
            flush_batch();
        }
    }
    // -- tail batch --
    if (!pending_.empty()) {
        flush_batch();
    }
}

// ---------------------------------------------------------------------------
// transient shells / export
// ---------------------------------------------------------------------------

py::list CppMCTSEngine::build_moves(int node_idx) {
    std::vector<std::pair<int, int>> steps;  // {action, mover color}
    int cur = node_idx;
    while (nodes_[cur].parent >= 0) {
        const int parent = nodes_[cur].parent;
        int action = -1;
        for (const auto& [a, ci] : nodes_[parent].children) {
            if (ci == cur) {
                action = a;
                break;
            }
        }
        steps.emplace_back(action, nodes_[parent].color);
        cur = parent;
    }
    const int size = boards_[nodes_[node_idx].board_index].size();
    py::list out;
    for (const auto& item : root_moves_) {
        out.append(item);
    }
    for (int i = static_cast<int>(steps.size()) - 1; i >= 0; --i) {
        const int action = steps[i].first;
        const int color = steps[i].second;
        if (action == size * size) {
            out.append(py::make_tuple(py::none(), color));
        } else {
            out.append(py::make_tuple(
                py::make_tuple(action / size, action % size), color));
        }
    }
    return out;
}

py::object CppMCTSEngine::get_shell(int node_idx) {
    // Cache hit requires a real (non-null) handle: the vector is resized with
    // default-constructed null py::objects, and is_none() is FALSE for a null
    // object -- only ptr() != nullptr distinguishes a built shell.
    if (static_cast<size_t>(node_idx) < shells_.size() &&
        shells_[node_idx].ptr() != nullptr) {
        return shells_[node_idx];
    }
    const CppNode& n = nodes_[node_idx];
    const Board& b = boards_[n.board_index];
    const int size = b.size();

    py::object last_captured = py::none();
    const int li = b.last_captured_index();
    if (li >= 0) {
        last_captured = py::make_tuple(li / size, li % size);
    }
    // Build Python lists explicitly (the same pattern the board binding uses):
    // py::cast of a std::vector from C++ resolves to the generic opaque
    // caster for lvalue references here, which yields a null handle.
    py::list state_list;
    for (const std::int8_t v : b.state()) {
        state_list.append(static_cast<int>(v));
    }
    py::object lm;
    if (n.legal_moves_known) {
        lm = py::list();
        for (const int a : n.legal_moves) {
            py::cast<py::list>(lm).append(a);
        }
    } else {
        lm = py::none();
    }
    py::object parent_shell =
        (n.parent >= 0) ? get_shell(n.parent) : py::object(py::none());

    py::object shell = make_node_(
        state_list, size, build_moves(node_idx),
        py::cast(b.pass_count()), last_captured, n.color, lm, n.prior,
        parent_shell, py::cast(n.visit_count), py::cast(n.value_sum));

    if (static_cast<size_t>(node_idx) >= shells_.size()) {
        shells_.resize(static_cast<size_t>(node_idx) + 1);
    }
    shells_[node_idx] = shell;
    return shell;
}

void CppMCTSEngine::set_stats(py::object shell, int node_idx) {
    shell.attr("visit_count") = py::cast(nodes_[node_idx].visit_count);
    shell.attr("value_sum") = py::cast(nodes_[node_idx].value_sum);
    shell.attr("virtual_loss") = py::cast(nodes_[node_idx].virtual_loss);
}

py::dict CppMCTSEngine::export_children(int node_idx) {
    py::dict out;
    for (const auto& [action, ci] : nodes_[node_idx].children) {
        py::object shell = get_shell(ci);
        set_stats(shell, ci);
        shell.attr("children") = export_children(ci);
        out[py::cast(action)] = shell;
    }
    return out;
}

void CppMCTSEngine::export_root() {
    set_stats(py_root_, root_idx_);
    py_root_.attr("children") = export_children(root_idx_);
}

}  // namespace omigamax
