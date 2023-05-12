import pytest

from chat_with_nerf.chat.grounder import ground


# Pytest fixtures for the ground method arguments
@pytest.fixture
def ground_text():
    return "example text"


@pytest.fixture
def visual_grounder(mocker):
    return mocker.Mock()


@pytest.fixture
def blip2captioner(mocker):
    return mocker.Mock()


# Test success case
def test_ground_success(mocker, ground_text, visual_grounder, blip2captioner):
    mocker.patch(
        "chat_with_nerf.chat.grounder.call_visual_grounder",
        return_value={"/path/to/image.jpg": "This is a caption."},
    )

    result = ground(ground_text, visual_grounder, blip2captioner)

    assert result == [("/path/to/image.jpg", "This is a caption.")]


# Test failure case (e.g., when call_visual_grounder raises an exception)
def test_ground_failure(mocker, ground_text, visual_grounder, blip2captioner):
    mocker.patch(
        "chat_with_nerf.chat.grounder.call_visual_grounder",
        side_effect=Exception("Something went wrong."),
    )

    with pytest.raises(Exception, match="Something went wrong."):
        ground(ground_text, visual_grounder, blip2captioner)
