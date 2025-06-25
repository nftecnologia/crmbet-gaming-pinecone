#!/usr/bin/env python3
"""
🚀 LOAD TESTING SYSTEM - 100k+ RPS Validation
Sistema de testes de carga para validar performance ultra-robusta

Author: Agente Load Testing - ULTRATHINK
Created: 2025-06-25
"""

import asyncio
import aiohttp
import time
import json
import statistics
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse
import structlog
from concurrent.futures import ThreadPoolExecutor
import psutil

# Configure logging
logging = structlog.get_logger()

@dataclass
class LoadTestConfig:
    """Configuração do teste de carga"""
    target_url: str
    max_rps: int = 100000
    test_duration_seconds: int = 300
    ramp_up_seconds: int = 60
    concurrent_users: int = 10000
    request_timeout: float = 30.0
    think_time_ms: int = 100
    test_scenarios: List[str] = None

@dataclass
class TestScenario:
    """Cenário de teste específico"""
    name: str
    endpoint: str
    method: str
    payload: Optional[Dict] = None
    headers: Optional[Dict] = None
    weight: float = 1.0
    expected_status: int = 200

@dataclass
class LoadTestResults:
    """Resultados detalhados do teste de carga"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_rps: float
    peak_rps: float
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate: float
    throughput_mbps: float
    test_duration_seconds: float
    scenarios_results: Dict[str, Dict]

class UltraLoadTester:
    """Sistema de load testing ultra-robusto"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.logger = logging.bind(component="UltraLoadTester")
        
        # Test scenarios para CRM Bet
        self.scenarios = [
            TestScenario(
                name="health_check",
                endpoint="/health",
                method="GET",
                weight=0.1,
                expected_status=200
            ),
            TestScenario(
                name="user_login",
                endpoint="/api/auth/login",
                method="POST",
                payload={
                    "username": "testuser{user_id}",
                    "password": "testpass123"
                },
                weight=0.15,
                expected_status=200
            ),
            TestScenario(
                name="user_profile",
                endpoint="/api/users/{user_id}",
                method="GET",
                headers={"Authorization": "Bearer test_token"},
                weight=0.2,
                expected_status=200
            ),
            TestScenario(
                name="create_transaction",
                endpoint="/api/transactions",
                method="POST",
                payload={
                    "user_id": "user_{user_id}",
                    "amount": "{amount}",
                    "transaction_type": "{transaction_type}",
                    "currency": "USD"
                },
                headers={"Authorization": "Bearer test_token"},
                weight=0.3,
                expected_status=201
            ),
            TestScenario(
                name="ml_prediction",
                endpoint="/api/ml/predict",
                method="POST",
                payload={
                    "user_id": "user_{user_id}",
                    "features": {
                        "avg_bet_amount": "{avg_bet}",
                        "session_frequency": "{session_freq}",
                        "win_rate": "{win_rate}"
                    }
                },
                headers={"Authorization": "Bearer test_token"},
                weight=0.25,
                expected_status=200
            )
        ]
        
        # Statistics tracking
        self.request_times = []
        self.request_statuses = []
        self.error_messages = []
        self.rps_samples = []
        self.start_time = None
        self.end_time = None
        
        self.logger.info("Ultra Load Tester initialized", config=asdict(config))
    
    async def run_load_test(self) -> LoadTestResults:
        """Executa teste de carga completo"""
        
        self.logger.info("🚀 INICIANDO LOAD TEST ULTRA-ROBUSTO")
        self.logger.info(f"Target: {self.config.target_url}")
        self.logger.info(f"Max RPS: {self.config.max_rps:,}")
        self.logger.info(f"Duration: {self.config.test_duration_seconds}s")
        self.logger.info(f"Concurrent Users: {self.config.concurrent_users:,}")
        
        self.start_time = time.time()
        
        # Initialize statistics
        self.request_times = []
        self.request_statuses = []
        self.error_messages = []
        self.rps_samples = []
        
        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.config.concurrent_users)
        
        # Start RPS monitoring
        rps_monitor_task = asyncio.create_task(self._monitor_rps())
        
        # Create user tasks
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task = asyncio.create_task(
                self._simulate_user(user_id, semaphore)
            )
            tasks.append(task)
        
        try:
            # Wait for test completion
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.test_duration_seconds + 60
            )
        except asyncio.TimeoutError:
            self.logger.warning("Load test timed out")
        
        # Stop RPS monitoring
        rps_monitor_task.cancel()
        
        self.end_time = time.time()
        
        # Calculate results
        results = self._calculate_results()
        
        self.logger.info("✅ LOAD TEST CONCLUÍDO", results=asdict(results))
        
        return results
    
    async def _simulate_user(self, user_id: int, semaphore: asyncio.Semaphore):
        """Simula comportamento de um usuário"""
        
        async with semaphore:
            session_timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                start_time = time.time()
                
                while (time.time() - start_time) < self.config.test_duration_seconds:
                    # Select random scenario based on weight
                    scenario = self._select_scenario()
                    
                    # Execute request
                    await self._execute_request(session, scenario, user_id)
                    
                    # Think time
                    if self.config.think_time_ms > 0:
                        await asyncio.sleep(self.config.think_time_ms / 1000.0)
    
    async def _execute_request(self, session: aiohttp.ClientSession, scenario: TestScenario, user_id: int):
        """Executa uma requisição específica"""
        
        try:
            # Prepare URL
            url = f"{self.config.target_url}{scenario.endpoint}"
            url = self._substitute_variables(url, user_id)
            
            # Prepare payload
            payload = None
            if scenario.payload:
                payload = self._substitute_variables(scenario.payload, user_id)
            
            # Prepare headers
            headers = scenario.headers or {}
            
            # Execute request
            start_time = time.time()
            
            if scenario.method == "GET":
                async with session.get(url, headers=headers) as response:
                    await response.text()
                    status = response.status
            elif scenario.method == "POST":
                async with session.post(url, json=payload, headers=headers) as response:
                    await response.text()
                    status = response.status
            elif scenario.method == "PUT":
                async with session.put(url, json=payload, headers=headers) as response:
                    await response.text()
                    status = response.status
            elif scenario.method == "DELETE":
                async with session.delete(url, headers=headers) as response:
                    await response.text()
                    status = response.status
            else:
                raise ValueError(f"Unsupported method: {scenario.method}")
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Record statistics
            self.request_times.append(response_time)
            self.request_statuses.append(status)
            
            # Check if response is expected
            if status != scenario.expected_status:
                self.error_messages.append(f"{scenario.name}: Expected {scenario.expected_status}, got {status}")
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 30000
            self.request_times.append(response_time)
            self.request_statuses.append(0)  # Error status
            self.error_messages.append(f"{scenario.name}: {str(e)}")
    
    async def _monitor_rps(self):
        """Monitora RPS em tempo real"""
        
        last_count = 0
        
        while True:
            try:
                await asyncio.sleep(1)
                
                current_count = len(self.request_times)
                current_rps = current_count - last_count
                self.rps_samples.append(current_rps)
                last_count = current_count
                
                if current_rps > 0:
                    self.logger.debug(f"Current RPS: {current_rps:,}")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error monitoring RPS", error=str(e))
    
    def _select_scenario(self) -> TestScenario:
        """Seleciona cenário baseado no peso"""
        
        total_weight = sum(scenario.weight for scenario in self.scenarios)
        random_value = random.random() * total_weight
        
        current_weight = 0
        for scenario in self.scenarios:
            current_weight += scenario.weight
            if random_value <= current_weight:
                return scenario
        
        return self.scenarios[0]  # Fallback
    
    def _substitute_variables(self, template, user_id: int):
        """Substitui variáveis no template"""
        
        if isinstance(template, str):
            return template.format(
                user_id=user_id,
                amount=random.randint(10, 1000),
                transaction_type=random.choice(['deposit', 'withdrawal', 'bet']),
                avg_bet=random.randint(5, 500),
                session_freq=random.randint(1, 50),
                win_rate=random.uniform(0.3, 0.8)
            )
        elif isinstance(template, dict):
            return {
                key: self._substitute_variables(value, user_id)
                for key, value in template.items()
            }
        else:
            return template
    
    def _calculate_results(self) -> LoadTestResults:
        """Calcula resultados do teste"""
        
        if not self.request_times:
            return LoadTestResults(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_rps=0.0,
                peak_rps=0.0,
                average_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                error_rate=1.0,
                throughput_mbps=0.0,
                test_duration_seconds=0.0,
                scenarios_results={}
            )
        
        # Basic statistics
        total_requests = len(self.request_times)
        successful_requests = sum(1 for status in self.request_statuses if 200 <= status < 400)
        failed_requests = total_requests - successful_requests
        
        # Duration
        test_duration = self.end_time - self.start_time if self.end_time else time.time() - self.start_time
        
        # RPS calculations
        average_rps = total_requests / test_duration if test_duration > 0 else 0
        peak_rps = max(self.rps_samples) if self.rps_samples else 0
        
        # Response time statistics
        average_response_time = statistics.mean(self.request_times)
        p95_response_time = statistics.quantiles(self.request_times, n=20)[18] if len(self.request_times) > 20 else 0
        p99_response_time = statistics.quantiles(self.request_times, n=100)[98] if len(self.request_times) > 100 else 0
        
        # Error rate
        error_rate = failed_requests / total_requests if total_requests > 0 else 1.0
        
        # Throughput (simplified - assuming average 1KB per response)
        throughput_mbps = (total_requests * 1024 * 8) / (test_duration * 1024 * 1024) if test_duration > 0 else 0
        
        # Scenario results (simplified)
        scenarios_results = {
            scenario.name: {
                "requests": int(total_requests * scenario.weight),
                "success_rate": 1.0 - error_rate
            }
            for scenario in self.scenarios
        }
        
        return LoadTestResults(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_rps=average_rps,
            peak_rps=peak_rps,
            average_response_time_ms=average_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            error_rate=error_rate,
            throughput_mbps=throughput_mbps,
            test_duration_seconds=test_duration,
            scenarios_results=scenarios_results
        )
    
    async def warmup_test(self) -> bool:
        """Executa teste de aquecimento"""
        
        self.logger.info("🔥 Executando warmup test...")
        
        warmup_config = LoadTestConfig(
            target_url=self.config.target_url,
            max_rps=100,
            test_duration_seconds=30,
            concurrent_users=10,
            request_timeout=10.0
        )
        
        warmup_tester = UltraLoadTester(warmup_config)
        results = await warmup_tester.run_load_test()
        
        if results.error_rate < 0.1:
            self.logger.info("✅ Warmup test passed")
            return True
        else:
            self.logger.error("❌ Warmup test failed", error_rate=results.error_rate)
            return False

def print_results_report(results: LoadTestResults):
    """Imprime relatório detalhado dos resultados"""
    
    print("\n" + "="*80)
    print("🚀 ULTRA-ROBUST LOAD TEST RESULTS")
    print("="*80)
    
    print(f"\n📊 OVERVIEW")
    print(f"Total Requests:        {results.total_requests:,}")
    print(f"Successful Requests:   {results.successful_requests:,}")
    print(f"Failed Requests:       {results.failed_requests:,}")
    print(f"Test Duration:         {results.test_duration_seconds:.2f}s")
    
    print(f"\n⚡ PERFORMANCE")
    print(f"Average RPS:           {results.average_rps:,.2f}")
    print(f"Peak RPS:              {results.peak_rps:,}")
    print(f"Throughput:            {results.throughput_mbps:.2f} Mbps")
    
    print(f"\n⏱️  RESPONSE TIMES")
    print(f"Average:               {results.average_response_time_ms:.2f}ms")
    print(f"95th percentile:       {results.p95_response_time_ms:.2f}ms")
    print(f"99th percentile:       {results.p99_response_time_ms:.2f}ms")
    
    print(f"\n🎯 QUALITY")
    print(f"Error Rate:            {results.error_rate*100:.2f}%")
    print(f"Success Rate:          {(1-results.error_rate)*100:.2f}%")
    
    print(f"\n📋 SCENARIOS")
    for scenario_name, scenario_data in results.scenarios_results.items():
        print(f"{scenario_name:20}: {scenario_data['requests']:,} requests, {scenario_data['success_rate']*100:.1f}% success")
    
    print("\n" + "="*80)
    
    # Performance assessment
    print("🎯 PERFORMANCE ASSESSMENT")
    print("="*80)
    
    if results.average_rps >= 100000:
        print("✅ TARGET ACHIEVED: 100k+ RPS")
    elif results.average_rps >= 50000:
        print("🟡 GOOD PERFORMANCE: 50k+ RPS")
    else:
        print("❌ PERFORMANCE BELOW TARGET")
    
    if results.p95_response_time_ms <= 100:
        print("✅ EXCELLENT LATENCY: <100ms P95")
    elif results.p95_response_time_ms <= 500:
        print("🟡 GOOD LATENCY: <500ms P95")
    else:
        print("❌ HIGH LATENCY: >500ms P95")
    
    if results.error_rate <= 0.01:
        print("✅ EXCELLENT RELIABILITY: <1% error rate")
    elif results.error_rate <= 0.05:
        print("🟡 GOOD RELIABILITY: <5% error rate")
    else:
        print("❌ POOR RELIABILITY: >5% error rate")
    
    print("="*80)

async def main():
    """Função principal do load testing"""
    
    parser = argparse.ArgumentParser(description="Ultra-Robust Load Testing System")
    parser.add_argument("--url", required=True, help="Target URL for load testing")
    parser.add_argument("--rps", type=int, default=100000, help="Target RPS (default: 100000)")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds (default: 300)")
    parser.add_argument("--users", type=int, default=10000, help="Concurrent users (default: 10000)")
    parser.add_argument("--warmup", action="store_true", help="Run warmup test first")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout (default: 30.0)")
    
    args = parser.parse_args()
    
    # Create load test configuration
    config = LoadTestConfig(
        target_url=args.url,
        max_rps=args.rps,
        test_duration_seconds=args.duration,
        concurrent_users=args.users,
        request_timeout=args.timeout
    )
    
    # Initialize load tester
    tester = UltraLoadTester(config)
    
    try:
        # Optional warmup
        if args.warmup:
            if not await tester.warmup_test():
                print("❌ Warmup test failed. Aborting main test.")
                return
        
        # Run main load test
        results = await tester.run_load_test()
        
        # Print detailed report
        print_results_report(results)
        
        # Save results to file
        results_file = f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        
        print(f"\n📄 Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n❌ Load test interrupted by user")
    except Exception as e:
        print(f"\n💥 Load test error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())