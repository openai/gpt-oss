from gpt_oss.responses_api.utils import stub_infer_next_token

def test_stub_infer_next_token_happy_path():
    # Test that the function returns tokens from the fake_tokens list
    result = stub_infer_next_token([])
    assert result in [200005, 35644, 200008, 1844, 31064, 25, 392, 4827, 382, 220, 17, 659, 16842, 12295, 81645, 51441, 6052, 17196, 314, 19, 9552, 238, 242, 200002]

def test_stub_infer_next_token_multiple_calls():
    # Test that multiple calls return different tokens from the list
    tokens = []
    for _ in range(5):
        tokens.append(stub_infer_next_token([]))
    assert len(tokens) == 5
    assert all(token in [200005, 35644, 200008, 1844, 31064, 25, 392, 4827, 382, 220, 17, 659, 16842, 12295, 81645, 51441, 6052, 17196, 314, 19, 9552, 238, 242, 200002] for token in tokens)

def test_stub_infer_next_token_resets_queue():
    # Test that the queue resets after depletion
    # Call the function enough times to deplete the queue
    with patch('time.sleep'):
        for _ in range(100):  # More than the length of fake_tokens
            stub_infer_next_token([])
    # The function should still work after reset
    result = stub_infer_next_token([])
    assert result in [200005, 35644, 200008, 1844, 31064, 25, 392, 4827, 382, 220, 17, 659, 16842, 12295, 81645, 51441, 6052, 17196, 314, 19, 9552, 238, 242, 200002]

def test_stub_infer_next_token_with_temperature():
    # Test that temperature parameter is accepted but not used in the stub
    result = stub_infer_next_token([], temperature=0.5)
    assert result in [200005, 35644, 200008, 1844, 31064, 25, 392, 4827, 382, 220, 17, 659, 16842, 12295, 81645, 51441, 6052, 17196, 314, 19, 9552, 238, 242, 200002]
