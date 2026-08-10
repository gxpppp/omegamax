// omigamax C++ rules core — bit-exact mirror of omigamax/rules/* (Python).
//
// Coordinates are (row, col), both 0-based, (0,0) top-left; colors are
// EMPTY=0 / BLACK=1 / WHITE=2. Semantics replicate (in lockstep):
//   board.py:124-157 play/pass with check_legal
//   legality.py + ko.py:14-28  is_legal (bounds/occupancy/suicide/simple-ko)
//   captures.py                capture / group removal
//   liberties.py:75-99         has_liberty (early exit)
//   scoring.py:27-104          territory + Tromp-Taylor score/winner/result
//
// No numpy/torch in this core; the pybind11 binding layer contains no rules
// logic — every rule lives here or in board.cpp.
#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace omigamax {

// Colors match omigamax/rules/liberties.py.
constexpr std::int8_t EMPTY = 0;
constexpr std::int8_t BLACK = 1;
constexpr std::int8_t WHITE = 2;

class IllegalMoveError : public std::invalid_argument {
public:
    explicit IllegalMoveError(const std::string& msg)
        : std::invalid_argument(msg) {}
};

// True if the group at (r, c) in a flat int8 state has at least one liberty.
// Mirrors liberties.py:75-99 (early-exits at the first liberty found; an
// empty point has no liberty).
bool has_liberty(const std::vector<std::int8_t>& state, int size, int r, int c);

// Ordered orthogonal neighbors of (r, c) within a `size` board, in the same
// order as liberties.py `neighbors()` (up, down, left, right).
void neighbors(int r, int c, int size,
               std::vector<std::pair<int, int>>& out);

class Board {
public:
    explicit Board(int size);

    // -- accessors -----------------------------------------------------
    int size() const { return size_; }
    int num_moves() const { return move_count_; }
    int pass_count() const { return pass_count_; }
    // Flat index of the single stone captured on the last move, or -1 (the
    // data simple-ko detection needs; mirrors Board.last_captured_point).
    int last_captured_index() const { return last_captured_; }
    const std::vector<std::int8_t>& state() const { return state_; }
    std::int8_t get(int r, int c) const;
    bool is_on_board(int r, int c) const;
    bool is_empty() const;

    // -- group / liberty queries (public API) --------------------------
    // Coordinates of the connected group containing (r, c) (empty -> empty).
    std::vector<std::pair<int, int>> group(int r, int c) const;
    // Empty points adjacent to the group containing (r, c).
    std::vector<std::pair<int, int>> liberties(int r, int c) const;
    int liberty_count(int r, int c) const;
    bool has_liberty(int r, int c) const;

    // -- terminal / scoring --------------------------------------------
    bool is_terminal() const { return pass_count_ >= 2; }
    // Map flat index -> owner for EMPTY points only: 0 neutral, 1/2 colors.
    std::map<int, int> territory() const;
    // Tromp-Taylor area score (black, white); komi added to white.
    std::pair<double, double> score(double komi = 7.5) const;
    // "B", "W" or "" for jigo (None in Python).
    std::string winner(double komi = 7.5) const;
    // SGF-style result, e.g. "B+3.5", "W+2", "Jigo".
    std::string result_string(double komi = 7.5) const;

    // -- play -----------------------------------------------------------
    // True if (r, c) is a legal point move for `color` (bounds, occupancy,
    // suicide prohibition and simple-ko). Pass is always legal.
    bool is_legal(int r, int c, int color) const;
    // Place a stone at (r, c) and remove captured opponent stones; returns
    // the number captured. Throws IllegalMoveError if check_legal and the
    // move is not legal (board left unchanged in that case).
    int play(int r, int c, int color, bool check_legal = true);
    // Play a pass; returns 0 (mirrors Board.play(None, color)).
    int pass_move(int color);
    // Ascending legal action indices for `color`: point indices (row*size+c)
    // in (row, col) order, then the pass index size*size (always legal).
    std::vector<int> legal_actions(int color) const;
    // Directly place a stone, resetting move history (pass_count_ /
    // last_captured_ / move_count_). Test/construction helper only.
    void set_stone(int r, int c, int color);

private:
    int idx(int r, int c) const { return r * size_ + c; }
    // bounds / occupancy / suicide only (no ko).
    bool is_legal_move(int r, int c, int color) const;
    // Remove adjacent liberty-less opponent groups; returns (total_removed,
    // captured group stone lists).
    std::pair<int, std::vector<std::vector<int>>> capture(int r, int c,
                                                          int color);

    int size_;
    std::vector<std::int8_t> state_;
    int move_count_ = 0;
    int pass_count_ = 0;
    int last_captured_ = -1;  // flat index or -1 (none)
};

}  // namespace omigamax
