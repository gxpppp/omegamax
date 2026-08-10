// pybind11 bindings for the omigamax C++ rules core.
//
// No rules logic lives here — every rule is in board.h / board.cpp; this
// layer only translates Python calls onto the C++ Board API.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "board.h"

namespace py = pybind11;

namespace {

py::list point_list(const std::vector<std::pair<int, int>>& pts) {
    py::list out;
    for (const auto& [r, c] : pts) {
        out.append(py::make_tuple(r, c));
    }
    return out;
}

// A flat int8/uint8 board state as passed by the Python reference (a list).
std::vector<std::int8_t> parse_state(const py::sequence& seq) {
    std::vector<std::int8_t> state;
    const py::ssize_t n = py::len(seq);
    state.reserve(static_cast<size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) {
        state.push_back(static_cast<std::int8_t>(py::cast<int>(seq[i])));
    }
    return state;
}

}  // namespace

PYBIND11_MODULE(omigamax_core, m) {
    m.doc() = "omigamax C++ rules core (bit-exact mirror of omigamax.rules)";

    // Free function mirroring liberties.py `has_liberty(state, size, r, c)`.
    m.def(
        "has_liberty",
        [](const py::sequence& state, int size, int r, int c) {
            return omigamax::has_liberty(parse_state(state), size, r, c);
        },
        py::arg("state"), py::arg("size"), py::arg("r"), py::arg("c"),
        "True if the group at (r, c) in a flat state has at least one "
        "liberty (mirrors liberties.has_liberty).");

    py::class_<omigamax::Board>(m, "CppBoard",
                                "Bit-exact C++ mirror of omigamax.rules.Board")
        .def(py::init<int>(), py::arg("size"),
             "Board with the given parameterized size (e.g. 9, 19).")
        .def_property_readonly("size", &omigamax::Board::size)
        .def_property_readonly("num_moves", &omigamax::Board::num_moves)
        .def_property_readonly("pass_count", &omigamax::Board::pass_count)
        .def_property_readonly(
            "last_captured_point",
            [](const omigamax::Board& b) {
                const int i = b.last_captured_index();
                if (i < 0) {
                    return py::object(py::none());
                }
                return py::object(py::make_tuple(i / b.size(), i % b.size()));
            },
            "Point (row, col) of a single-stone capture on the last move, "
            "or None.")
        .def("state",
             [](const omigamax::Board& b) {
                 py::list out;
                 for (const std::int8_t v : b.state()) {
                     out.append(static_cast<int>(v));
                 }
                 return out;
             },
             "Copy of the flat board state (index r*size + c).")
        .def("get",
             [](const omigamax::Board& b, int r, int c) {
                 return static_cast<int>(b.get(r, c));
             },
             py::arg("r"), py::arg("c"))
        .def("is_on_board", &omigamax::Board::is_on_board, py::arg("r"),
             py::arg("c"))
        .def("is_empty", &omigamax::Board::is_empty)
        .def("group",
             [](const omigamax::Board& b, int r, int c) {
                 return point_list(b.group(r, c));
             },
             py::arg("r"), py::arg("c"))
        .def("liberties",
             [](const omigamax::Board& b, int r, int c) {
                 return point_list(b.liberties(r, c));
             },
             py::arg("r"), py::arg("c"))
        .def("liberty_count", &omigamax::Board::liberty_count, py::arg("r"),
             py::arg("c"))
        .def("has_liberty", &omigamax::Board::has_liberty, py::arg("r"),
             py::arg("c"))
        .def("is_terminal", &omigamax::Board::is_terminal)
        .def("territory",
             [](const omigamax::Board& b) {
                 py::dict out;
                 for (const auto& [i, owner] : b.territory()) {
                     const int r = i / b.size();
                     const int c = i % b.size();
                     out[py::make_tuple(r, c)] =
                         (owner == 0) ? py::object(py::none())
                                      : py::cast(owner);
                 }
                 return out;
             },
             "Map each empty point to its territory owner (None neutral).")
        .def("score", &omigamax::Board::score, py::arg("komi") = 7.5)
        .def("winner",
             [](const omigamax::Board& b, double komi) {
                 const std::string w = b.winner(komi);
                 return w.empty() ? py::object(py::none()) : py::cast(w);
             },
             py::arg("komi") = 7.5)
        .def("result_string", &omigamax::Board::result_string,
             py::arg("komi") = 7.5)
        .def("is_legal", &omigamax::Board::is_legal, py::arg("r"), py::arg("c"),
             py::arg("color"))
        .def("play",
             [](omigamax::Board& b, int r, int c, int color,
                bool check_legal) {
                 try {
                     return b.play(r, c, color, check_legal);
                 } catch (const omigamax::IllegalMoveError& e) {
                     throw py::value_error(e.what());
                 }
             },
             py::arg("r"), py::arg("c"), py::arg("color"),
             py::arg("check_legal") = true,
             "Place a stone (or raise ValueError if illegal); returns the "
             "number of stones captured.")
        .def("pass_move", &omigamax::Board::pass_move, py::arg("color"))
        .def("legal_actions",
             [](const omigamax::Board& b, int color) {
                 int eff = color;
                 if (eff < 0) {
                     eff = (b.num_moves() % 2 == 0) ? omigamax::BLACK
                                                    : omigamax::WHITE;
                 }
                 return b.legal_actions(eff);
             },
             py::arg("color") = -1,
             "Ascending legal action indices (points then the pass index "
             "size*size); color None derives from move-count parity.")
        .def("set_stone", &omigamax::Board::set_stone, py::arg("r"),
             py::arg("c"), py::arg("color"),
             "Directly place a stone, resetting move history "
             "(test/construction helper).");
}
