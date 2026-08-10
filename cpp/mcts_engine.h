// AlphaGo-Zero MCTS search engine (C++ tree) with the transient-shell leaf
// protocol -- bit-exact mirror of omigamax.mcts.mcts.run_search on the
// batched-evaluator path.
//
// Design (Oracle-adjudicated, see .omo/plans):
//   * the C++ tree owns every node and board exclusively; there is NO Python
//     mirror and NO bidirectional sync during the search;
//   * when a leaf is submitted to the batched evaluator, a TRANSIENT Python
//     Node shell is built (board snapshot + threaded color + legal_moves +
//     parent chain + the attribute set the evaluator reads); after the flush
//     the C++ engine discards the shell and backprops into its own nodes;
//   * identity matching: the shell object submitted is the same object the
//     evaluator returns (the flush order is verified with `is`), so the
//     mcts.py:536-541 protocol holds;
//   * the caller's Python root is imported once (existing subtree + stats)
//     and the final tree is materialized back onto it by export().
//
// Bit-exactness vs Python select_child/expand/backup:
//   * UCB is computed in IEEE double with the SAME operation order
//     ``q + c_puct * prior * sqrt(N_parent) / (1 + N_eff)``, ties break to the
//     lowest action index (children iterated in ascending action order);
//   * expand deep-copies the parent board per legal child and plays the action
//     with check_legal=False (board.cpp is differential-tested bit-exact);
//   * backup negates the value at every level, ``value_sum += v`` in double.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "board.h"

namespace omigamax {

namespace py = pybind11;

class CppMCTSEngine {
public:
    // py_root: the caller's Python Node root (its existing subtree is
    //          imported); evaluator_: batched evaluator (submit/flush);
    //          make_node: Python helper that builds a transient Node shell;
    //          noisy_prior: {action: float} root Dirichlet override or None.
    CppMCTSEngine(py::object py_root, py::object evaluator, int simulations,
                  double c_puct, double komi, int virtual_loss, int batch_size,
                  py::object make_node, py::object noisy_prior);

    // Run the search loop (Python run_search semantics, batched path).
    void run();
    // Materialize the C++ tree back onto the caller's Python root.
    void export_root();

private:
    struct CppNode {
        int parent = -1;
        int board_index = -1;
        std::vector<std::pair<int, int>> children;  // {action, node}, ascending
        double prior = 0.0;
        int visit_count = 0;
        double value_sum = 0.0;
        int virtual_loss = 0;
        bool expanded = false;
        int color = 0;
        std::vector<int> legal_moves;
        bool legal_moves_known = false;
    };
    struct PendingItem {
        int node_idx;
        std::vector<int> path;  // root .. leaf
        py::object shell;       // the submitted transient shell (identity)
    };

    // -- pools -----------------------------------------------------------
    int alloc_node();
    int clone_board(int board_index);
    const Board& board(int node_idx) const {
        return boards_[nodes_[node_idx].board_index];
    }
    int child_of(int node_idx, int action) const;

    // -- search ----------------------------------------------------------
    int select_child(int node_idx) const;
    void expand(int node_idx, const float* prior);
    void backup(const std::vector<int>& path, double v);
    void flush_batch();
    bool is_pending(int node_idx) const;
    void ensure_legal_moves(int node_idx);

    // -- transient shells / export --------------------------------------
    py::object get_shell(int node_idx);  // cached per node
    py::list build_moves(int node_idx);
    void set_stats(py::object shell, int node_idx);
    py::dict export_children(int node_idx);

    // -- import ----------------------------------------------------------
    int import_node(const py::object& pynode, int parent_idx, int parent_color);

    // -- python state ----------------------------------------------------
    py::object py_root_;
    py::object evaluator_;
    py::object make_node_;
    py::list root_moves_;  // copy of the Python root's board.moves
    std::map<int, double> noisy_;

    // -- config ----------------------------------------------------------
    int simulations_;
    double c_puct_;
    double komi_;
    int virtual_loss_;
    int batch_size_;

    // -- tree ------------------------------------------------------------
    std::vector<CppNode> nodes_;
    std::vector<Board> boards_;
    int root_idx_ = -1;
    std::vector<py::object> shells_;  // node idx -> shell (py::none = unbuilt)
    std::vector<PendingItem> pending_;
};

}  // namespace omigamax
