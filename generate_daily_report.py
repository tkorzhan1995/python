#!/usr/bin/env python
"""
Finance Order-to-Cash Reconciliation Report Generator with Python Decorators

Usage:
    python generate_daily_report.py --data-dir ./data --output-dir ./output --variance-threshold 1.0
"""
import pandas as pd
import click
from datetime import date
import os
from pathlib import Path
import functools
import time
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Decorators
def timer(func):
    """Decorator to measure execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def log_calls(func):
    """Decorator to log function calls with arguments."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        params = []
        if args:
            params.extend([f"arg{i}={arg}" for i, arg in enumerate(args[:2])])
        if kwargs:
            params.extend([f"{k}={v}" for k, v in list(kwargs.items())[:3]])
        params_str = ", ".join(params)
        logger.info(f"Calling {func.__name__}({params_str})")

        try:
            result = func(*args, **kwargs)
            logger.info(f"Function '{func.__name__}' completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function '{func.__name__}' failed with error: {str(e)}")
            raise
    return wrapper

def validate_inputs(func):
    """Decorator to validate input parameters."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        data_dir = kwargs.get('data_dir', args[0] if args else '../data')
        if not Path(data_dir).exists():
            logger.warning(f"Data directory '{data_dir}' does not exist")

        threshold = kwargs.get('variance_threshold', args[2] if len(args) > 2 else 1.0)
        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError(f"variance_threshold must be a positive number, got {threshold}")

        return func(*args, **kwargs)
    return wrapper

def retry(max_attempts=3, delay=1, backoff=2):
    """Decorator to retry functions with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except (FileNotFoundError, pd.errors.EmptyDataError) as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Function '{func.__name__}' failed after {max_attempts} attempts")
                        raise e

                    logger.warning(f"Attempt {attempts} failed: {str(e)}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                except Exception as e:
                    logger.error(f"Non-recoverable error in '{func.__name__}': {str(e)}")
                    raise e
            return None
        return wrapper
    return decorator

@timer
@log_calls
@validate_inputs
@retry(max_attempts=2, delay=0.5)
def load_data_files(data_path):
    """Load all required data files with retry logic."""
    orders = pd.read_csv(data_path / "orders.csv")
    orders = orders[orders["order_id"] != "order_id"]

    refunds = pd.read_csv(data_path / "refunds.csv")
    psp = pd.read_csv(data_path / "psp_settlements.csv")
    gl = pd.read_csv(data_path / "gl_entries.csv")

    logger.info(f"Loaded {len(orders)} orders, {len(refunds)} refunds, {len(psp)} PSP settlements, {len(gl)} GL entries")
    return orders, refunds, psp, gl

@timer
@log_calls
def process_data(orders, gl, variance_threshold):
    """Process and reconcile orders with GL data."""
    orders = orders.rename(columns={"net_amount": "expected_net"})
    gl = gl.rename(columns={"amount": "gl_amount"})

    orders.columns = orders.columns.str.strip().str.lower()
    gl.columns = gl.columns.str.strip().str.lower()

    orders = orders.rename(columns={"net_amount": "expected_net"})
    gl = gl.rename(columns={"reference": "order_id"})

    gl["debit"] = pd.to_numeric(gl["debit"], errors="coerce")
    gl["credit"] = pd.to_numeric(gl["credit"], errors="coerce")
    gl["signed_amount"] = gl["debit"].fillna(0) - gl["credit"].fillna(0)

    gl_order = (gl.groupby("order_id", as_index=False)
                  .agg(gl_amount=("signed_amount", "sum")))

    merged = orders.merge(gl_order, on="order_id", how="left")
    merged["variance"] = merged["expected_net"] - merged["gl_amount"]

    exceptions = merged[
        merged["gl_amount"].isna() | (merged["variance"].abs() > variance_threshold)
    ]
    return exceptions

@timer
@log_calls
def create_exception_report(exceptions):
    """Create formatted exception report."""
    report = pd.DataFrame({
        "report_date": date.today(),
        "exception_type": "GL_MISMATCH", 
        "order_id": exceptions["order_id"],
        "variance": exceptions["variance"],
        "status": "OPEN",
        "priority": "HIGH"
    })
    return report

@timer
@log_calls
@validate_inputs
def generate_reconciliation_report(data_dir='../data', output_dir='../output', variance_threshold=1.0):
    """Generate daily reconciliation report with configurable directories."""

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        orders, refunds, psp, gl = load_data_files(data_path)
        exceptions = process_data(orders, gl, variance_threshold)
        report = create_exception_report(exceptions)

        output_file = output_path / "daily_exception_report.csv"
        report.to_csv(output_file, index=False)

        logger.info(f"Generated {len(report)} exceptions")
        logger.info(f"Report saved to: {output_file}")

        return report

    except Exception as e:
        logger.error(f"Failed to generate reconciliation report: {str(e)}")
        return None

@click.command()
@click.option('--data-dir', '-d', default='../data', 
              help='Directory containing input data files')
@click.option('--output-dir', '-o', default='../output', 
              help='Directory for output files')
@click.option('--variance-threshold', '-t', default=1.0, type=float,
              help='Variance threshold for exception reporting')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
def main(data_dir, output_dir, variance_threshold, verbose):
    """Generate daily reconciliation report with configurable directories."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    generate_reconciliation_report(data_dir, output_dir, variance_threshold)

if __name__ == "__main__":
    main()
