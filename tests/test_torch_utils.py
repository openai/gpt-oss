"""Unit tests for torch distributed initialization utilities."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSuppressOutput:
    """Test suppress_output functionality."""

    def test_suppress_output_rank_zero_prints(self, capsys):
        """Test that rank 0 prints normally."""
        from gpt_oss.torch.utils import suppress_output
        
        suppress_output(0)
        print("test message")
        captured = capsys.readouterr()
        
        assert "test message" in captured.out

    def test_suppress_output_non_zero_rank_suppressed(self, capsys):
        """Test that non-zero ranks are suppressed."""
        from gpt_oss.torch.utils import suppress_output
        
        suppress_output(1)
        print("should not appear")
        captured = capsys.readouterr()
        
        assert "should not appear" not in captured.out

    def test_suppress_output_force_prints_any_rank(self, capsys):
        """Test that force=True prints from any rank."""
        from gpt_oss.torch.utils import suppress_output
        
        suppress_output(2)
        print("forced message", force=True)
        captured = capsys.readouterr()
        
        assert "rank #2:" in captured.out
        assert "forced message" in captured.out


class TestInitDistributed:
    """Test init_distributed functionality."""

    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_cuda_not_available_raises(self, mock_dist, mock_torch):
        """Test that RuntimeError is raised when CUDA is not available."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = False
        
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            init_distributed()

    @patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"})
    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_single_gpu_success(self, mock_dist, mock_torch):
        """Test successful initialization with single GPU."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_device = Mock()
        mock_torch.device.return_value = mock_device
        
        device = init_distributed()
        
        assert device == mock_device
        mock_torch.cuda.set_device.assert_called_once_with(0)
        mock_dist.init_process_group.assert_not_called()

    @patch.dict(os.environ, {"WORLD_SIZE": "4", "RANK": "2"})
    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_multi_gpu_success(self, mock_dist, mock_torch):
        """Test successful initialization with multiple GPUs."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        mock_device = Mock()
        mock_torch.device.return_value = mock_device
        mock_dist.is_initialized.return_value = False
        
        device = init_distributed()
        
        assert device == mock_device
        mock_torch.cuda.set_device.assert_called_once_with(2)
        mock_dist.init_process_group.assert_called_once_with(
            backend="nccl", init_method="env://", world_size=4, rank=2
        )

    @patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "5"})
    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_rank_exceeds_devices_raises(self, mock_dist, mock_torch):
        """Test that RuntimeError is raised when rank exceeds available devices."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        
        with pytest.raises(RuntimeError, match="Rank 5 exceeds available CUDA devices"):
            init_distributed()

    @patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"})
    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_device_access_failure_raises(self, mock_dist, mock_torch):
        """Test that RuntimeError is raised when device access fails."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_properties.side_effect = RuntimeError("Device error")
        mock_dist.is_initialized.return_value = False
        
        with pytest.raises(RuntimeError, match="Failed to access CUDA device"):
            init_distributed()

    @patch.dict(os.environ, {"WORLD_SIZE": "4", "RANK": "1"})
    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_nccl_warmup_success(self, mock_dist, mock_torch):
        """Test NCCL warmup executes for multi-GPU setup."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        mock_device = Mock()
        mock_torch.device.return_value = mock_device
        mock_tensor = Mock()
        mock_torch.ones.return_value = mock_tensor
        mock_dist.is_initialized.return_value = False
        
        device = init_distributed()
        
        # Verify NCCL warmup was attempted
        mock_torch.ones.assert_called_once_with(1, device=mock_device)
        mock_dist.all_reduce.assert_called_once_with(mock_tensor)
        mock_torch.cuda.synchronize.assert_called_once_with(mock_device)

    @patch.dict(os.environ, {"WORLD_SIZE": "4", "RANK": "0"})
    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_nccl_warmup_failure_raises(self, mock_dist, mock_torch):
        """Test that RuntimeError is raised when NCCL warmup fails."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        mock_device = Mock()
        mock_torch.device.return_value = mock_device
        mock_torch.ones.return_value = Mock()
        mock_dist.all_reduce.side_effect = RuntimeError("NCCL error")
        mock_dist.is_initialized.return_value = True
        
        with pytest.raises(RuntimeError, match="Failed to initialize distributed communication"):
            init_distributed()
        
        # Verify cleanup was attempted
        mock_dist.destroy_process_group.assert_called_once()

    @patch.dict(os.environ, {})
    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_default_env_vars(self, mock_dist, mock_torch):
        """Test initialization with default environment variables."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_device = Mock()
        mock_torch.device.return_value = mock_device
        
        device = init_distributed()
        
        # Should default to WORLD_SIZE=1, RANK=0
        assert device == mock_device
        mock_torch.cuda.set_device.assert_called_once_with(0)
        mock_dist.init_process_group.assert_not_called()

    @patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"})
    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_cleanup_on_exception(self, mock_dist, mock_torch):
        """Test that process group is cleaned up on exception."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.set_device.side_effect = RuntimeError("Set device failed")
        mock_dist.is_initialized.return_value = True
        
        with pytest.raises(RuntimeError):
            init_distributed()
        
        # Verify cleanup was attempted
        mock_dist.destroy_process_group.assert_called_once()

    @patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"})
    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_cleanup_error_suppressed(self, mock_dist, mock_torch):
        """Test that cleanup errors are suppressed."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.set_device.side_effect = RuntimeError("Set device failed")
        mock_dist.is_initialized.return_value = True
        mock_dist.destroy_process_group.side_effect = RuntimeError("Cleanup failed")
        
        # Should raise original error, not cleanup error
        with pytest.raises(RuntimeError, match="Set device failed"):
            init_distributed()


class TestDistributedEnvironment:
    """Test distributed environment variable handling."""

    @patch.dict(os.environ, {"WORLD_SIZE": "8", "RANK": "3"})
    @patch('gpt_oss.torch.utils.torch')
    @patch('gpt_oss.torch.utils.dist')
    def test_init_distributed_respects_env_vars(self, mock_dist, mock_torch):
        """Test that environment variables are correctly parsed."""
        from gpt_oss.torch.utils import init_distributed
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 8
        mock_device = Mock()
        mock_torch.device.return_value = mock_device
        mock_dist.is_initialized.return_value = False
        
        device = init_distributed()
        
        # Verify rank 3 was used
        mock_torch.cuda.set_device.assert_called_once_with(3)
        mock_dist.init_process_group.assert_called_once_with(
            backend="nccl", init_method="env://", world_size=8, rank=3
        )
