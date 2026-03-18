# Copyright (c) 2025 Tencent Inc. All rights reserved.

import logging
import torch
import gcore_web_sdk
from typing import Dict, Any, Optional, TYPE_CHECKING

# Module-level private variable for singleton instance
# TYPE_CHECKING used for type hints without circular imports
if TYPE_CHECKING:
    _sdk_instance: Optional["GCoreSDK"] = None
else:
    _sdk_instance = None

class GCoreSDK:
    """GCore SDK wrapper with distributed training support (process-level singleton)."""
    
    def __init__(
        self,
        org: str,
        project: str,
        name: str,
        main_process: bool = False,
        log_level: int = logging.ERROR,
        device: Optional[str] = None
    ):
        """
        Initialize GCore SDK.
        
        Args:
            org: Organization name
            project: Project name
            name: Experiment name
            main_process: Whether this is the main process (rank 0)
            log_level: Logging level
            device: Device for distributed communication (auto-detected if None)
        """
        self.org = org
        self.project = project
        self.name = name
        self.log_level = log_level
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.run_id: Optional[int] = None
        self.main_process = main_process
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize SDK and synchronize run ID across distributed processes."""
        if self.main_process:
            self._init_main_process()
        
        self._synchronize_run_id()
        
        if self.run_id is None:
            raise RuntimeError("Failed to initialize GCore SDK: run_id is None")
    
    def _init_main_process(self) -> None:
        """Initialize SDK on main process (rank 0)."""
        run = gcore_web_sdk.init(
            org=self.org,
            project=self.project,
            name=self.name,
            log_level=self.log_level,
        )
        if not run:
            raise RuntimeError("gcore_web_sdk initialization failed")
        self.run_id = run.id
        logging.info(f"GCore SDK initialized with run_id={self.run_id}")
    
    def _synchronize_run_id(self) -> None:
        """Synchronize run ID across distributed processes."""
        # Create tensor for broadcast (main process has valid ID)
        run_id_tensor = torch.tensor(
            [self.run_id] if self.run_id is not None else [-1],
            dtype=torch.int,
            device=self.device
        )
        
        # Broadcast from main process (rank 0)
        torch.distributed.broadcast(run_id_tensor, src=0)
        
        # Update run_id from broadcasted value
        self.run_id = run_id_tensor.item()
    
    def is_stopping_task(self) -> bool:
        """
        Check if the task is being stopped.
        
        Returns:
            True if task is stopping, False otherwise
        """
        if self.run_id is None:
            logging.warning("Cannot check stopping status: SDK not initialized")
            return False
            
        try:
            run = gcore_web_sdk.try_get_run(self.run_id)
            if run is None:
                logging.warning(f"Failed to get run status for run_id={self.run_id}")
                return False
            return run.status == gcore_web_sdk.RunStatus.Stopping.value
        except Exception as e:
            logging.error(f"Error checking task status: {str(e)}")
            return False
    
    def report(self, detail: Dict[str, Any]) -> None:
        """
        Report metrics to GCore.
        
        Args:
            detail: Dictionary of metrics to report
        """
        if self.run_id is None:
            return
            
        try:
            gcore_web_sdk.report(detail)
        except Exception as e:
            logging.error(f"Failed to report metrics: {str(e)}")

def init_sdk(
    org: str,
    project: str,
    name: str,
    main_process: bool = False,
    log_level: int = logging.ERROR,
    device: Optional[str] = None
) -> GCoreSDK:
    """
    Initialize GCore SDK (singleton pattern).
    
    Note: Can only be called once per process, after distributed initialization.
    
    Args:
        org: Organization name
        project: Project name
        name: Experiment name
        main_process: Whether this is the main process (rank 0)
        log_level: Logging level
        device: Device for distributed communication
    
    Returns:
        GCoreSDK instance (always the same instance per process)
    
    Raises:
        RuntimeError: If SDK is already initialized
    """
    global _sdk_instance
    
    if _sdk_instance is not None:
        raise RuntimeError(
            "GCore SDK is already initialized. "
            "Use get_sdk() to access the existing instance."
        )
    
    _sdk_instance = GCoreSDK(
        org=org,
        project=project,
        name=name,
        main_process=main_process,
        log_level=log_level,
        device=device
    )
    return _sdk_instance

def get_sdk() -> GCoreSDK:
    """
    Get the initialized GCore SDK instance.
    
    Returns:
        GCoreSDK instance
    
    Raises:
        RuntimeError: If SDK has not been initialized
    """
    global _sdk_instance
    
    if _sdk_instance is None:
        raise RuntimeError(
            "GCore SDK has not been initialized. "
            "Call init_sdk() first with appropriate parameters."
        )
    
    return _sdk_instance

# Legacy API compatibility (optional)
def gwsdk_init(*args, **kwargs) -> None:
    """Legacy API compatibility function for initialization."""
    init_sdk(*args, **kwargs)

def gwsdk_is_stopping_task() -> bool:
    """Legacy API compatibility function to check if task is stopping."""
    return get_sdk().is_stopping_task()

def gwsdk_report(detail: Dict[str, Any]) -> None:
    """Legacy API compatibility function to report metrics."""
    get_sdk().report(detail)