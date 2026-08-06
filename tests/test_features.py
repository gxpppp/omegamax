"""TDD tests for the 17-plane feature encoding and policy index mapping (todo 7).

Per the plan (todo 7) and the AGZ paper (Nature 550, 2017, Methods), the input
feature stack is ``st = [Xt, Yt, Xt-1, Yt-1, ..., Xt-7, Yt-7, C]`` -- exactly
17 planes, *no* constant-1 plane:

  * plane 2t   = X_{t-t}   : current player's stones, position t moves ago
                              (t = 0 is the current position, most recent first)
  * plane 2t+1 = Y_{t-t}   : opponent's stones, position t moves ago
  * plane 16   = C         : colour to play -- all 1.0 if black, all 0.0 if white

Missing history (fewer than 8 positions at game start) is zero-filled.

Policy indices (consistent with todo 6's model, ``pass`` at index ``N**2``):
  * point (r, c) <-> index ``r * board_size + c``  (single convention)
  * pass is index ``board_size ** 2``

Covers: encode shape/dtype, plane content, history shift / zero-fill /
truncation, binary & finite values, index mapping round-trip, encode_batch,
decode_policy masking, and end-to-end network integration.
"""

import numpy as np
import pytest
import torch

from omigamax.network.features import (
    decode_policy,
    encode,
    encode_batch,
    index_to_point,
    is_pass,
    pass_index,
    point_to_index,
)
from omigamax.network.model import create_model, policy_logit_count
from omigamax.rules import BLACK, EMPTY, WHITE, Board

TOTAL_PLANES = 17  # 8 history x 2 colours + colour-to-play plane


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_state(size, stones):
    """Build a flat board state (list of ``size*size`` colour codes).

    ``stones`` maps colour -> list of (row, col) coordinates.
    """
    state = [EMPTY] * (size * size)
    for color, points in stones.items():
        for r, c in points:
            state[r * size + c] = color
    return state


def assert_all_zero(plane):
    assert np.all(plane == 0.0), "plane is not all-zero"


# ---------------------------------------------------------------------------
# encode: shape / dtype / colour plane (empty history)
# ---------------------------------------------------------------------------

def test_encode_empty_history_default_board_shape_and_dtype():
    """The plan's acceptance shape: encode([], 1) -> (17, 19, 19) float32."""
    a = encode([], BLACK)
    assert a.shape == (17, 19, 19)
    assert a.dtype == np.float32


def test_encode_empty_history_all_stone_planes_zero():
    a = encode([], BLACK)
    assert_all_zero(a[0:16])


def test_encode_color_plane_black_is_all_ones():
    a = encode([], BLACK)
    assert np.all(a[16] == 1.0)


def test_encode_color_plane_white_is_all_zeros():
    a = encode([], WHITE)
    assert np.all(a[16] == 0.0)


# ---------------------------------------------------------------------------
# encode: plane content (current position in plane 0 / 1)
# ---------------------------------------------------------------------------

def test_encode_black_stone_in_own_plane_when_black_to_play():
    state = make_state(9, {BLACK: [(3, 3)]})
    a = encode([state], BLACK, board_size=9)
    assert a.shape == (17, 9, 9)
    assert a[0][3, 3] == 1.0  # X_t: current player (black) stone
    assert a[1][3, 3] == 0.0  # Y_t: opponent (white) has nothing there
    assert a[2][3, 3] == 0.0  # no older history
    assert np.all(a[16] == 1.0)  # black to play


def test_encode_black_stone_in_opponent_plane_when_white_to_play():
    state = make_state(9, {BLACK: [(3, 3)]})
    a = encode([state], WHITE, board_size=9)
    assert a[0][3, 3] == 0.0  # X_t: current player (white) has nothing there
    assert a[1][3, 3] == 1.0  # Y_t: opponent (black) stone
    assert np.all(a[16] == 0.0)  # white to play


# ---------------------------------------------------------------------------
# encode: exact AGZ interleaved plane ordering
# ---------------------------------------------------------------------------

def test_encode_interleaved_plane_layout_matches_agz():
    """Verify the paper ordering [Xt, Yt, Xt-1, Yt-1, Xt-2, Yt-2, ..., C]."""
    size = 9
    state_t = make_state(size, {BLACK: [(0, 0)], WHITE: [(1, 1)]})  # current
    state_t1 = make_state(size, {BLACK: [(0, 0)]})                   # 1 move ago
    state_t2 = make_state(size, {})                                  # 2 moves ago
    a = encode([state_t, state_t1, state_t2], BLACK, board_size=size)
    assert a.shape == (17, 9, 9)

    # plane 0 = X_t : current player's stones at the current position
    assert a[0][0, 0] == 1.0
    assert a[0][1, 1] == 0.0
    # plane 1 = Y_t : opponent's stones at the current position
    assert a[1][1, 1] == 1.0
    assert a[1][0, 0] == 0.0
    # plane 2 = X_{t-1} : current player's stones 1 move ago
    assert a[2][0, 0] == 1.0
    # plane 3 = Y_{t-1} : opponent had no stones 1 move ago
    assert_all_zero(a[3])
    # plane 4 = X_{t-2}, plane 5 = Y_{t-2} : empty position
    assert_all_zero(a[4])
    assert_all_zero(a[5])
    # planes 6..15 : no history at all
    assert_all_zero(a[6:16])
    # plane 16 = C : black to play
    assert np.all(a[16] == 1.0)


def test_encode_history_shift_captured_stone_only_in_older_plane():
    """A stone captured on the last move must appear in the older plane,
    not in the current-position plane."""
    size = 5
    state_t = make_state(size, {BLACK: [(0, 0)]})  # white stone at (0,1) captured
    state_t1 = make_state(size, {BLACK: [(0, 0)], WHITE: [(0, 1)]})  # 1 ago
    state_t2 = make_state(size, {BLACK: [(0, 0)]})  # 2 ago: white not yet played
    a = encode([state_t, state_t1, state_t2], BLACK, board_size=size)

    # white stone is gone from the current position -> not in Y_t (plane 1)
    assert a[1][0, 1] == 0.0
    # but it was present 1 move ago -> in Y_{t-1} (plane 3)
    assert a[3][0, 1] == 1.0
    # black stone at (0,0) persists across all positions
    assert a[0][0, 0] == 1.0
    assert a[2][0, 0] == 1.0
    assert a[4][0, 0] == 1.0


def test_stone_moves_to_older_plane_after_new_move():
    """After a new move, the previous position shifts to the '1 move ago'
    plane: the X-plane for that position moves from index 0 to index 2."""
    b = Board(9)
    b.play((4, 4), BLACK)   # move 1
    b.play((3, 3), WHITE)   # move 2
    pos_now = b.state
    b1 = Board(9)
    b1.play((4, 4), BLACK)
    pos_one_ago = b1.state

    a = encode([pos_now, pos_one_ago], BLACK, board_size=9)
    # black stone (4,4) played at move 1: present in current X_t (plane 0)
    assert a[0][4, 4] == 1.0
    # ... and, since the position is 1 move ago, also in X_{t-1} (plane 2)
    assert a[2][4, 4] == 1.0
    # white stone (3,3) played at move 2 is only in the current position
    assert a[1][3, 3] == 1.0
    assert a[3][3, 3] == 0.0


# ---------------------------------------------------------------------------
# encode: insufficient history, over-long history, values
# ---------------------------------------------------------------------------

def test_encode_insufficient_history_zero_filled():
    """With only 1 position supplied, planes for t=1..7 stay all-zero."""
    state = make_state(19, {BLACK: [(0, 0)]})
    a = encode([state], BLACK)
    assert a[0][0, 0] == 1.0
    assert_all_zero(a[1:16])


def test_encode_history_longer_than_8_truncated():
    """Only the 8 most recent positions are encoded; older ones are dropped."""
    size = 10
    # 10 snapshots with distinct single-stone marks at (i, i), newest first
    states = [make_state(size, {BLACK: [(i, i)]}) for i in range(10)]
    a = encode(states, BLACK, board_size=size)
    # snapshot 0 (most recent) -> X_t plane 0
    assert a[0][0, 0] == 1.0
    # snapshot 7 (8th most recent) -> X_{t-7} plane 14
    assert a[14][7, 7] == 1.0
    # snapshots 8 and 9 fall outside the 8-step window and must not leak
    for t in range(8):
        assert a[2 * t][8, 8] == 0.0
        assert a[2 * t][9, 9] == 0.0


def test_encode_values_binary_and_finite():
    size = 5
    state = make_state(size, {BLACK: [(1, 1)], WHITE: [(2, 2)]})
    a = encode([state, state], BLACK, board_size=size)
    assert np.isfinite(a).all()
    assert set(np.unique(a).tolist()) <= {0.0, 1.0}


def test_encode_infers_board_size_from_snapshot():
    state = make_state(9, {BLACK: [(0, 0)]})
    a = encode([state], BLACK)  # no board_size argument
    assert a.shape == (17, 9, 9)


def test_encode_accepts_board_object():
    b = Board(9)
    b.play((5, 5), BLACK)
    a = encode([b], WHITE)
    assert a.shape == (17, 9, 9)
    assert a[1][5, 5] == 1.0  # black stone as opponent (white to play)


def test_encode_rejects_invalid_colour():
    state = make_state(9, {})
    with pytest.raises(ValueError):
        encode([state], 0)  # EMPTY is not a side to move


# ---------------------------------------------------------------------------
# encode_batch
# ---------------------------------------------------------------------------

def test_encode_batch_shape():
    size = 9
    s1 = make_state(size, {BLACK: [(0, 0)]})
    s2 = make_state(size, {BLACK: [(1, 1)]})
    s3 = make_state(size, {WHITE: [(2, 2)]})
    batch = encode_batch([[s1], [s2], [s3]], [BLACK, BLACK, WHITE], board_size=size)
    assert batch.shape == (3, 17, 9, 9)
    assert batch.dtype == np.float32
    assert batch[0][0][0, 0] == 1.0
    assert batch[1][0][1, 1] == 1.0
    assert batch[2][0][2, 2] == 1.0  # white (current player) stone -> X_t
    assert batch[2][1][2, 2] == 0.0  # not in opponent plane
    assert np.all(batch[2][16] == 0.0)


def test_encode_batch_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        encode_batch([make_state(9, {})], [BLACK, WHITE], board_size=9)


# ---------------------------------------------------------------------------
# action-index mapping (point <-> index, pass at N**2)
# ---------------------------------------------------------------------------

def test_point_index_roundtrip():
    size = 19
    for r, c in [(0, 0), (0, 18), (9, 9), (18, 0), (18, 18), (5, 12)]:
        i = point_to_index(r, c, size)
        assert i == r * size + c
        assert index_to_point(i, size) == (r, c)
    assert point_to_index(0, 0, 9) == 0
    assert point_to_index(8, 8, 9) == 80


def test_pass_index_is_board_squared():
    assert pass_index(19) == 361
    assert pass_index(9) == 81
    assert is_pass(361, 19)
    assert not is_pass(360, 19)
    assert is_pass(81, 9)
    assert not is_pass(0, 9)


def test_pass_index_matches_model_policy_logit_count():
    for size in (9, 19):
        assert policy_logit_count(size) == pass_index(size) + 1


def test_index_to_point_out_of_range_raises():
    with pytest.raises(ValueError):
        index_to_point(-1, 9)
    with pytest.raises(ValueError):
        index_to_point(81, 9)  # pass index is not a point


def test_point_to_index_out_of_board_raises():
    with pytest.raises(ValueError):
        point_to_index(9, 0, 9)
    with pytest.raises(ValueError):
        point_to_index(0, -1, 9)


# ---------------------------------------------------------------------------
# decode_policy: legal-move distribution from logits
# ---------------------------------------------------------------------------

def test_decode_policy_shape_and_normalised():
    b = Board(9)  # empty board, black to move
    p = decode_policy(np.zeros(82), b)
    assert p.shape == (82,)
    assert p.dtype == np.float32
    assert np.isclose(p.sum(), 1.0)
    assert (p >= 0).all()


def test_decode_policy_masks_illegal_moves():
    b = Board(9)
    b.play((0, 0), BLACK)  # black stone at (0,0); it is now white's turn
    logits = np.zeros(82)
    logits[0] = 100.0  # strongly favour (0,0), which is illegal for white
    p = decode_policy(logits, b)  # colour derived from move count (1 -> white)
    assert p[0] == 0.0  # illegal move gets exactly zero probability
    assert np.isclose(p.sum(), 1.0)
    assert p[pass_index(9)] > 0.0  # pass is always legal


def test_decode_policy_side_to_move_derivation():
    b = Board(9)
    # 0 moves played -> black to move
    p = decode_policy(np.zeros(82), b)
    assert p[point_to_index(0, 0, 9)] > 0.0
    b.play((0, 0), BLACK)   # 1 move -> white to move
    p = decode_policy(np.zeros(82), b)
    assert p[0] == 0.0  # (0,0) occupied and illegal for white
    b.play((1, 1), WHITE)   # 2 moves -> black to move again
    p = decode_policy(np.zeros(82), b)
    assert p[0] == 0.0
    assert p[point_to_index(1, 1, 9)] == 0.0
    assert p[point_to_index(2, 2, 9)] > 0.0


def test_decode_policy_accepts_torch_logits():
    b = Board(19)
    logits = torch.zeros(362)
    p = decode_policy(logits, b)
    assert np.isclose(p.sum(), 1.0)
    assert p.shape == (362,)


def test_decode_policy_combined_moves_excluded_when_occupied():
    b = Board(9)
    b.play((4, 4), BLACK)   # 1 move, white to play
    p = decode_policy(np.zeros(82), b)
    assert p[point_to_index(4, 4, 9)] == 0.0


# ---------------------------------------------------------------------------
# network integration: encoded tensor through the policy-value model
# ---------------------------------------------------------------------------

def test_encoded_tensor_through_b10c128_network():
    torch.manual_seed(0)
    b = Board(19)
    b.play((9, 9), BLACK)
    b.play((3, 3), WHITE)
    b.play((15, 15), BLACK)
    features = encode([b.state], WHITE)  # (17, 19, 19) float32
    assert features.shape == (17, 19, 19)
    x = torch.from_numpy(features).unsqueeze(0)  # (1, 17, 19, 19)
    assert x.shape == (1, 17, 19, 19)

    model = create_model(10, 128, 19)
    model.eval()
    with torch.no_grad():
        policy, value = model(x)
    assert policy.shape == (1, 362)
    assert value.shape == (1, 1)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()

    # decode the network's raw policy logits into a legal-move distribution
    p = decode_policy(policy[0], b)
    assert np.isclose(p.sum(), 1.0)
    assert p.shape == (362,)
