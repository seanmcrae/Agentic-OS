"""System monitoring utilities for Agentic OS
Provides resource tracking and system health monitoring."""
from typing import Dict, Any
import psutil
import time
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    timestamp: datetime

class SystemMonitor:
    """Monitors system resources and health"""
    
    def __init__(self):
        self.logger = logging.getLogger("system_monitor")
        self._metrics_history: List[SystemMetrics] = []
        self._started = False

    async def start(self):
        """Start system monitoring"""
        if self._started:
            return

        self._started = True
        self.logger.info("System monitoring started")

    async def stop(self):
        """Stop system monitoring"""
        if not self._started:
            return

        self._started = False
        self.logger.info("System monitoring stopped")

    async def get_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            metrics = SystemMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                disk_usage=psutil.disk_usage('/').percent,
                network_io=dict(psutil.net_io_counters()._asdict()),
                timestamp=datetime.utcnow()
            )
            
            self._metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {str(e)}")
            raise

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            metrics = self.get_metrics()
            
            # Define health thresholds
            cpu_threshold = 80
            memory_threshold = 85
            disk_threshold = 90
            
            # Check resource usage
            status = {
                "status": "healthy",
                "warnings": [],
                "metrics": {
                    "cpu": metrics.cpu_percent,
                    "memory": metrics.memory_percent,
                    "disk": metrics.disk_usage
                },
                "timestamp": metrics.timestamp
            }
            
            # Add warnings for high resource usage
            if metrics.cpu_percent > cpu_threshold:
                status["warnings"].append(f"High CPU usage: {metrics.cpu_percent}%")
                
            if metrics.memory_percent > memory_threshold:
                status["warnings"].append(f"High memory usage: {metrics.memory_percent}%")
                
            if metrics.disk_usage > disk_threshold:
                status["warnings"].append(f"High disk usage: {metrics.disk_usage}%")
                
            if status["warnings"]:
                status["status"] = "warning"
                
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get health status: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow()
            }

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get detailed resource usage information"""
        try:
            return {
                "cpu": {
                    "percent": psutil.cpu_percent(interval=1, percpu=True),
                    "count": psutil.cpu_count(),
                    "frequency": dict(psutil.cpu_freq()._asdict())
                },
                "memory": dict(psutil.virtual_memory()._asdict()),
                "disk": {
                    "usage": dict(psutil.disk_usage('/')._asdict()),
                    "io": dict(psutil.disk_io_counters()._asdict())
                },
                "network": {
                    "io": dict(psutil.net_io_counters()._asdict()),
                    "connections": len(psutil.net_connections())
                },
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {str(e)}")
            raise"
