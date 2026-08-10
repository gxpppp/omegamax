// Implementation of the omigamax C++ rules core (see board.h).
//
// Every rule below replicates the Python reference exactly 鈥?the diff test
// tests/test_cpp_rules_diff.py asserts bit-exact equality against
// omigamax/rules/* on every move of 1000+ games plus boundary positions.

#include "board.h"

#include <cmath>

namespace omigamax {

namespace {

// Full group of the stone at (r, c) as flat indices; empty point -> empty.
// Mirrors liberties.py `group()` (DFS stack flood fill).
std::vector<int> group_of(const std::vector<std::int8_t>& state, int size,
                          int r, int c) {
    std::vector<int> result;
    const std::int8_t color = state[r * size + c];
    if (color == EMPTY) {
        return result;
    }
    std::vector<char> visited(static_cast<size_t>(size * size), 0);
    std::vector<int> stack;
    stack.reserve(static_cast<size_t>(size * size));
    stack.push_back(r * size + c);
    while (!stack.empty()) {
        const int cur = stack.back();
        stack.pop_back();
        if (visited[cur]) {
            continue;
        }
        visited[cur] = 1;
        result.push_back(cur);
        const int cr = cur / size;
        const int cc = cur % size;
        std::vector<std::pair<int, int>> nbrs;
        neighbors(cr, cc, size, nbrs);
        for (const auto& [nr, nc] : nbrs) {
            const int ni = nr * size + nc;
            if (state[ni] == color && !visited[ni]) {
                stack.push_back(ni);
            }
        }
    }
    return result;
}

// True if any opponent group adjacent to (r, c) has no liberties on `state`
// (which already has `color` placed at (r, c)). Same truthiness as
// captures.py `captured_groups()` being non-empty.
bool has_capturable_group(const std::vector<std::int8_t>& state, int size,
                          int r, int c, int color) {
    const std::int8_t opp = (color == BLACK) ? WHITE : BLACK;
    std::vector<char> seen(static_cast<size_t>(size * size), 0);
    std::vector<std::pair<int, int>> nbrs;
    neighbors(r, c, size, nbrs);
    for (const auto& [nr, nc] : nbrs) {
        const int ni = nr * size + nc;
        if (seen[ni]) {
            continue;
        }
        if (state[ni] != opp) {
            continue;
        }
        if (has_liberty(state, size, nr, nc)) {
            continue;
        }
        const std::vector<int> stones = group_of(state, size, nr, nc);
        for (const int s : stones) {
            seen[s] = 1;
        }
        return true;
    }
    return false;
}

// Replicates Python f"{x:g}" for the only values TT scoring can produce:
// non-negative integers and half-integers (0.0 / 0.5 / 1.0 / 1.5 / ...).
std::string fmt_margin(double d) {
    const double rounded = std::floor(d);
    if (d == rounded) {
        return std::to_string(static_cast<long long>(rounded));
    }
    return std::to_string(static_cast<long long>(rounded)) + ".5";
}

}  // namespace

void neighbors(int r, int c, int size,
               std::vector<std::pair<int, int>>& out) {
    out.clear();
    if (r > 0) {
        out.emplace_back(r - 1, c);
    }
    if (r < size - 1) {
        out.emplace_back(r + 1, c);
    }
    if (c > 0) {
        out.emplace_back(r, c - 1);
    }
    if (c < size - 1) {
        out.emplace_back(r, c + 1);
    }
}

bool has_liberty(const std::vector<std::int8_t>& state, int size, int r,
                 int c) {
    const std::int8_t color = state[r * size + c];
    if (color == EMPTY) {
        return false;
    }
    std::vector<char> visited(static_cast<size_t>(size * size), 0);
    std::vector<int> stack;
    stack.reserve(static_cast<size_t>(size * size));
    stack.push_back(r * size + c);
    while (!stack.empty()) {
        const int cur = stack.back();
        stack.pop_back();
        if (visited[cur]) {
            continue;
        }
        visited[cur] = 1;
        const int cr = cur / size;
        const int cc = cur % size;
        std::vector<std::pair<int, int>> nbrs;
        neighbors(cr, cc, size, nbrs);
        for (const auto& [nr, nc] : nbrs) {
            const int ni = nr * size + nc;
            if (state[ni] == EMPTY) {
                return true;
            }
            if (state[ni] == color && !visited[ni]) {
                stack.push_back(ni);
            }
        }
    }
    return false;
}

Board::Board(int size)
    : size_(size), state_(static_cast<size_t>(size * size), EMPTY) {
    if (size <= 0) {
        throw std::invalid_argument("board size must be positive");
    }
}

std::int8_t Board::get(int r, int c) const { return state_[idx(r, c)]; }

bool Board::is_on_board(int r, int c) const {
    return 0 <= r && r < size_ && 0 <= c && c < size_;
}

bool Board::is_empty() const {
    for (const std::int8_t v : state_) {
        if (v != EMPTY) {
            return false;
        }
    }
    return true;
}

std::vector<std::pair<int, int>> Board::group(int r, int c) const {
    std::vector<std::pair<int, int>> out;
    const std::vector<int> stones = group_of(state_, size_, r, c);
    out.reserve(stones.size());
    for (const int i : stones) {
        out.emplace_back(i / size_, i % size_);
    }
    return out;
}

std::vector<std::pair<int, int>> Board::liberties(int r, int c) const {
    std::vector<std::pair<int, int>> result;
    const std::vector<int> stones = group_of(state_, size_, r, c);
    std::vector<char> seen(static_cast<size_t>(size_ * size_), 0);
    std::vector<std::pair<int, int>> nbrs;
    for (const int i : stones) {
        const int gr = i / size_;
        const int gc = i % size_;
        neighbors(gr, gc, size_, nbrs);
        for (const auto& [nr, nc] : nbrs) {
            const int ni = nr * size_ + nc;
            if (state_[ni] == EMPTY && !seen[ni]) {
                seen[ni] = 1;
                result.emplace_back(nr, nc);
            }
        }
    }
    return result;
}

int Board::liberty_count(int r, int c) const {
    return static_cast<int>(liberties(r, c).size());
}

bool Board::has_liberty(int r, int c) const {
    return omigamax::has_liberty(state_, size_, r, c);
}

std::map<int, int> Board::territory() const {
    std::map<int, int> own;
    std::vector<char> visited(static_cast<size_t>(size_ * size_), 0);
    for (int start = 0; start < size_ * size_; ++start) {
        if (visited[start] || state_[start] != EMPTY) {
            continue;
        }
        std::vector<int> region;
        std::vector<std::int8_t> bordering;
        std::vector<int> stack;
        stack.push_back(start);
        while (!stack.empty()) {
            const int cur = stack.back();
            stack.pop_back();
            if (visited[cur]) {
                continue;
            }
            visited[cur] = 1;
            region.push_back(cur);
            const int r = cur / size_;
            const int c = cur % size_;
            std::vector<std::pair<int, int>> nbrs;
            neighbors(r, c, size_, nbrs);
            for (const auto& [nr, nc] : nbrs) {
                const int ni = nr * size_ + nc;
                if (state_[ni] == EMPTY) {
                    stack.push_back(ni);
                } else {
                    bordering.push_back(state_[ni]);
                }
            }
        }
        // Region is a set-collection of bordering colors; owner is the single
        // distinct color, or neutral (0) when bordered by both / none.
        const std::int8_t first = bordering.empty() ? EMPTY : bordering[0];
        int owner = 0;
        if (!bordering.empty()) {
            bool single = true;
            for (const std::int8_t b : bordering) {
                if (b != first) {
                    single = false;
                    break;
                }
            }
            if (single) {
                owner = first;
            }
        }
        for (const int i : region) {
            own[i] = owner;
        }
    }
    return own;
}

std::pair<double, double> Board::score(double komi) const {
    int black_stones = 0;
    int white_stones = 0;
    for (const std::int8_t v : state_) {
        if (v == BLACK) {
            ++black_stones;
        } else if (v == WHITE) {
            ++white_stones;
        }
    }
    int neutral = 0;
    int black_territory = 0;
    int white_territory = 0;
    for (const auto& [i, owner] : territory()) {
        (void)i;
        if (owner == BLACK) {
            ++black_territory;
        } else if (owner == WHITE) {
            ++white_territory;
        } else {
            ++neutral;
        }
    }
    const double half = static_cast<double>(neutral) / 2.0;
    // Same left-to-right evaluation as scoring.py:64-84; every term is
    // exactly representable (integers + 0.5 increments + komi), so the
    // result is bit-exact vs Python's float arithmetic.
    const double black_total =
        static_cast<double>(black_stones + black_territory) + half;
    const double white_total =
        static_cast<double>(white_stones + white_territory) + half + komi;
    return {black_total, white_total};
}

std::string Board::winner(double komi) const {
    const auto [black, white] = score(komi);
    if (black > white) {
        return "B";
    }
    if (white > black) {
        return "W";
    }
    return "";
}

std::string Board::result_string(double komi) const {
    const auto [black, white] = score(komi);
    if (black > white) {
        return "B+" + fmt_margin(black - white);
    }
    if (white > black) {
        return "W+" + fmt_margin(white - black);
    }
    return "Jigo";
}

bool Board::is_legal_move(int r, int c, int color) const {
    if (!is_on_board(r, c)) {
        return false;
    }
    const int i = idx(r, c);
    if (state_[i] != EMPTY) {
        return false;
    }
    std::vector<std::pair<int, int>> nbrs;
    neighbors(r, c, size_, nbrs);
    for (const auto& [nr, nc] : nbrs) {
        if (state_[nr * size_ + nc] == EMPTY) {
            return true;  // fast path: stone keeps at least one liberty
        }
    }
    // Every neighbor occupied: legal iff the move captures at least one
    // opponent stone or the placed stone joins a group that keeps a liberty.
    // Simulate the placement on a copy (Python restores in `finally`).
    std::vector<std::int8_t> tmp = state_;
    tmp[i] = static_cast<std::int8_t>(color);
    if (has_capturable_group(tmp, size_, r, c, color)) {
        return true;
    }
    return omigamax::has_liberty(tmp, size_, r, c);
}

bool Board::is_legal(int r, int c, int color) const {
    if (!is_legal_move(r, c, color)) {
        return false;
    }
    // Simple-ko: the move may not immediately retake the single stone
    // captured on the opponent's last move (ko.py:14-28).
    if (last_captured_ >= 0 && idx(r, c) == last_captured_) {
        return false;
    }
    return true;
}

std::pair<int, std::vector<std::vector<int>>> Board::capture(int r, int c,
                                                             int color) {
    const std::int8_t opp = (color == BLACK) ? WHITE : BLACK;
    std::vector<std::vector<int>> groups;
    std::vector<char> seen(static_cast<size_t>(size_ * size_), 0);
    std::vector<std::pair<int, int>> nbrs;
    neighbors(r, c, size_, nbrs);
    for (const auto& [nr, nc] : nbrs) {
        const int ni = nr * size_ + nc;
        if (seen[ni]) {
            continue;
        }
        if (state_[ni] != opp) {
            continue;
        }
        if (omigamax::has_liberty(state_, size_, nr, nc)) {
            continue;
        }
        std::vector<int> stones = group_of(state_, size_, nr, nc);
        for (const int s : stones) {
            seen[s] = 1;
        }
        groups.push_back(std::move(stones));
    }
    int total = 0;
    for (const auto& g : groups) {
        for (const int s : g) {
            if (state_[s] != EMPTY) {
                state_[s] = EMPTY;
                ++total;
            }
        }
    }
    return {total, std::move(groups)};
}

int Board::play(int r, int c, int color, bool check_legal) {
    if (check_legal && !is_legal(r, c, color)) {
        throw IllegalMoveError("illegal move (" + std::to_string(r) + ", " +
                               std::to_string(c) + ") for color " +
                               std::to_string(color));
    }
    pass_count_ = 0;
    state_[idx(r, c)] = static_cast<std::int8_t>(color);
    auto [removed, groups] = capture(r, c, color);
    if (removed == 1 && !groups.empty()) {
        // Single-stone capture -> record its point for simple-ko. The group
        // has exactly one stone, so the point is unambiguous.
        last_captured_ = groups[0][0];
    } else {
        last_captured_ = -1;
    }
    ++move_count_;
    return removed;
}

int Board::pass_move(int color) {
    (void)color;
    ++pass_count_;
    last_captured_ = -1;
    ++move_count_;
    return 0;
}

std::vector<int> Board::legal_actions(int color) const {
    std::vector<int> out;
    for (int r = 0; r < size_; ++r) {
        for (int c = 0; c < size_; ++c) {
            if (is_legal(r, c, color)) {
                out.push_back(idx(r, c));
            }
        }
    }
    out.push_back(size_ * size_);  // pass index (always legal)
    return out;
}

void Board::set_stone(int r, int c, int color) {
    state_[idx(r, c)] = static_cast<std::int8_t>(color);
    pass_count_ = 0;
    last_captured_ = -1;
    move_count_ = 0;
}

void Board::load_position(const std::vector<std::int8_t>& state,
                          int move_count, int pass_count,
                          int last_captured_index) {
    if (static_cast<int>(state.size()) != size_ * size_) {
        throw std::invalid_argument(
            "state length does not match board size");
    }
    state_ = state;
    move_count_ = move_count;
    pass_count_ = pass_count;
    last_captured_ = last_captured_index;
}

}  // namespace omigamax
