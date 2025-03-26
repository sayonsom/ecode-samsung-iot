#!/usr/bin/env python3
# data_quality_monitor.py
# Data Quality Monitoring Script for Energy Metrics

import os
import sys
import logging
import pandas as pd
import numpy as np
import argparse
import json
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("data_quality_monitor")

# Energy source categories
INTERMITTENT_SOURCES = {"SUN", "WND"}
CRITICAL_SOURCES = {"COL", "NUC", "NG"}
ALL_SOURCES = {"BAT", "COL", "GEO", "NG", "NUC", "OES", "OIL", "OTH", "PS", "SNB", "SUN", "UES", "WAT", "WND"}

# Default weights for different data quality issues
DEFAULT_WEIGHTS = {
    "missing_data": 1.0,
    "flatline": 0.8,
    "negative_values": 0.9,
    "outliers": 0.7,
    "zero_values": {
        "INTERMITTENT": 0.3,  # Lower penalty for intermittent sources like wind/solar
        "CRITICAL": 0.9,      # High penalty for critical sources
        "OTHER": 0.5          # Medium penalty for other sources
    }
}

# Alerting thresholds
ALERT_THRESHOLDS = {
    "overall_score": 5,  # Overall quality score below this triggers an alert
    "missing_data": 7,   # Missing data score below this triggers an alert
    "flatline": 6,       # Flatline score below this triggers an alert
    "negative_values": 8, # Negative values score below this triggers an alert
    "outliers": 5,       # Outliers score below this triggers an alert
    "zero_values": 6     # Zero values score below this triggers an alert
}

def connect_to_database(db_url: Optional[str] = None) -> 'sqlalchemy.engine.Engine':
    """
    Connect to the database.
    
    Args:
        db_url: Database URL (optional)
        
    Returns:
        SQLAlchemy engine
    """
    if db_url is None:
        db_url = os.getenv("SQLITE_URL", "sqlite:///data/energy_metrics.db")
    
    logger.info(f"Connecting to database: {db_url}")
    return create_engine(db_url)

def get_regions(engine) -> List[str]:
    """
    Get all available regions from the database.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        List of region codes
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT region FROM energy_metrics"))
            regions = [row[0] for row in result]
        logger.info(f"Found {len(regions)} regions: {', '.join(regions)}")
        return regions
    except Exception as e:
        logger.error(f"Error getting regions: {e}")
        return []

def get_energy_data(engine, region: str, days: int = 7) -> pd.DataFrame:
    """
    Get energy data for a region for the specified period.
    
    Args:
        engine: SQLAlchemy engine
        region: Region code
        days: Number of days of data to retrieve
        
    Returns:
        DataFrame with energy data
    """
    try:
        cutoff = datetime.now() - timedelta(days=days)
        query = text("""
        SELECT * FROM energy_metrics 
        WHERE region = :region 
        AND timestamp >= :cutoff 
        ORDER BY timestamp
        """)
        
        df = pd.read_sql_query(
            query, 
            engine, 
            params={"region": region, "cutoff": cutoff}
        )
        
        if df.empty:
            logger.warning(f"No data found for region {region} in the last {days} days")
            return pd.DataFrame()
        
        # Convert timestamp to datetime if it's not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Retrieved {len(df)} records for region {region} from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    except Exception as e:
        logger.error(f"Error getting data for region {region}: {e}")
        return pd.DataFrame()

def check_missing_data(df: pd.DataFrame, sources: List[str]) -> Dict[str, Dict]:
    """
    Check for missing data in the dataframe.
    
    Args:
        df: DataFrame with energy data
        sources: List of energy sources to check
        
    Returns:
        Dictionary with results of missing data check
    """
    if df.empty:
        return {'overall': 0.0, 'details': {source: 0.0 for source in sources}}
    
    results = {}
    
    for source in sources:
        if source in df.columns:
            missing_pct = df[source].isna().mean() * 100
            results[source] = {
                'missing_pct': missing_pct,
                'score': 10 - (missing_pct / 10)  # 10 if no missing, 0 if all missing
            }
        else:
            results[source] = {
                'missing_pct': 100.0,
                'score': 0.0
            }
    
    # Calculate overall score for missing data
    overall_score = sum(source_result['score'] for source_result in results.values()) / len(results)
    
    return {
        'overall': overall_score,
        'details': results
    }

def check_flatline(df: pd.DataFrame, sources: List[str], window: int = 6) -> Dict[str, Dict]:
    """
    Check for flatlined data (unchanged for several consecutive periods).
    
    Args:
        df: DataFrame with energy data
        sources: List of energy sources to check
        window: Number of consecutive periods to check for flatlines
        
    Returns:
        Dictionary with results of flatline check
    """
    if df.empty or len(df) < window:
        return {'overall': 0.0, 'details': {source: 0.0 for source in sources}}
    
    results = {}
    
    for source in sources:
        if source in df.columns:
            # Count periods where value doesn't change
            diff = df[source].diff()
            # Create a series that's 1 when there's no change, 0 when there is
            no_change = (diff == 0)
            
            # Count consecutive flatlines
            flatline_count = 0
            current_count = 0
            
            for value in no_change:
                if value:
                    current_count += 1
                else:
                    if current_count >= window:
                        flatline_count += current_count - window + 1
                    current_count = 0
                    
            # Check if there's a flatline at the end of the series
            if current_count >= window:
                flatline_count += current_count - window + 1
                
            # Calculate percentage of data points in flatlines
            flatline_pct = (flatline_count / len(df)) * 100 if len(df) > 0 else 0
            
            # Score is inversely proportional to flatline percentage
            results[source] = {
                'flatline_pct': flatline_pct,
                'score': 10 - (flatline_pct / 10)  # 10 if no flatlines, 0 if 100% flatlines
            }
        else:
            results[source] = {
                'flatline_pct': 0.0,
                'score': 10.0
            }
    
    # Calculate overall score for flatlines
    overall_score = sum(source_result['score'] for source_result in results.values()) / len(results)
    
    return {
        'overall': overall_score,
        'details': results
    }

def check_negative_values(df: pd.DataFrame, sources: List[str]) -> Dict[str, Dict]:
    """
    Check for negative values in the dataframe.
    
    Args:
        df: DataFrame with energy data
        sources: List of energy sources to check
        
    Returns:
        Dictionary with results of negative value check
    """
    if df.empty:
        return {'overall': 0.0, 'details': {source: 0.0 for source in sources}}
    
    results = {}
    
    for source in sources:
        if source in df.columns:
            negative_pct = (df[source] < 0).mean() * 100
            results[source] = {
                'negative_pct': negative_pct,
                'score': 10 - negative_pct  # 10 if no negatives, 0 if 10% or more are negative
            }
        else:
            results[source] = {
                'negative_pct': 0.0,
                'score': 10.0
            }
    
    # Calculate overall score for negative values
    overall_score = sum(source_result['score'] for source_result in results.values()) / len(results)
    
    return {
        'overall': overall_score,
        'details': results
    }

def check_outliers(df: pd.DataFrame, sources: List[str], std_threshold: float = 2.0) -> Dict[str, Dict]:
    """
    Check for outliers (values that are more than std_threshold standard deviations from the mean).
    
    Args:
        df: DataFrame with energy data
        sources: List of energy sources to check
        std_threshold: Threshold for outliers in terms of standard deviations
        
    Returns:
        Dictionary with results of outlier check
    """
    if df.empty:
        return {'overall': 0.0, 'details': {source: 0.0 for source in sources}}
    
    results = {}
    
    for source in sources:
        if source in df.columns and len(df[source].dropna()) > 1:
            mean = df[source].mean()
            std = df[source].std()
            
            if std == 0:
                # All values are the same, no outliers but potential flatline issue
                results[source] = {
                    'outlier_pct': 0.0,
                    'score': 10.0
                }
                continue
                
            # Count outliers
            outliers = (abs(df[source] - mean) > std_threshold * std)
            outlier_pct = outliers.mean() * 100
            
            # Score is inversely proportional to outlier percentage (capped at a reasonable level)
            results[source] = {
                'outlier_pct': outlier_pct,
                'score': max(0, 10 - (outlier_pct / 2))  # 10 if no outliers, 0 if 20% or more are outliers
            }
        else:
            results[source] = {
                'outlier_pct': 0.0,
                'score': 10.0
            }
    
    # Calculate overall score for outliers
    overall_score = sum(source_result['score'] for source_result in results.values()) / len(results)
    
    return {
        'overall': overall_score,
        'details': results
    }

def check_zero_values(df: pd.DataFrame, sources: List[str], historical_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
    """
    Check for zero values, accounting for historical patterns.
    
    Args:
        df: DataFrame with energy data
        sources: List of energy sources to check
        historical_df: Historical data (if available) to determine usual patterns
        
    Returns:
        Dictionary with results of zero values check
    """
    if df.empty:
        return {'overall': 0.0, 'details': {source: 0.0 for source in sources}}
    
    results = {}
    sources_normally_zero = set()
    
    # Identify sources that are normally zero based on historical data
    if historical_df is not None and not historical_df.empty:
        for source in sources:
            if source in historical_df.columns:
                # If 95% or more of historical values are zero, consider this normal
                if (historical_df[source] == 0).mean() >= 0.95:
                    sources_normally_zero.add(source)
    
    for source in sources:
        if source in df.columns:
            zero_pct = (df[source] == 0).mean() * 100
            
            # Don't penalize for zeros if this source is normally zero
            if source in sources_normally_zero:
                score = 10.0
            else:
                # Different penalty based on source type
                if source in INTERMITTENT_SOURCES:
                    # Less penalty for intermittent sources (solar can be zero at night, wind can be zero on calm days)
                    score = 10 - (zero_pct * DEFAULT_WEIGHTS['zero_values']['INTERMITTENT'] / 10)
                elif source in CRITICAL_SOURCES:
                    # High penalty for critical sources that shouldn't be zero
                    score = 10 - (zero_pct * DEFAULT_WEIGHTS['zero_values']['CRITICAL'] / 10)
                else:
                    # Medium penalty for other sources
                    score = 10 - (zero_pct * DEFAULT_WEIGHTS['zero_values']['OTHER'] / 10)
            
            results[source] = {
                'zero_pct': zero_pct,
                'normally_zero': source in sources_normally_zero,
                'score': max(0, score)
            }
        else:
            results[source] = {
                'zero_pct': 0.0,
                'normally_zero': False,
                'score': 10.0
            }
    
    # Calculate overall score for zero values
    overall_score = sum(source_result['score'] for source_result in results.values()) / len(results)
    
    return {
        'overall': overall_score,
        'details': results
    }

def calculate_overall_quality_score(check_results: Dict[str, Dict], weights: Dict = None) -> int:
    """
    Calculate overall data quality score from all check results.
    
    Args:
        check_results: Results from all quality checks
        weights: Weights for each check type (optional)
        
    Returns:
        Integer score from 0 to 10
    """
    if weights is None:
        weights = {
            'missing_data': 1.0,
            'flatline': 1.0,
            'negative_values': 1.0,
            'outliers': 0.8,
            'zero_values': 1.0
        }
    
    weighted_scores = []
    for check_type, results in check_results.items():
        if check_type in weights and 'overall' in results:
            weighted_scores.append(results['overall'] * weights[check_type])
    
    if not weighted_scores:
        return 0
    
    # Calculate weighted average and round to integer
    weighted_avg = sum(weighted_scores) / sum(weights.values())
    return round(weighted_avg)

def analyze_region_data_quality(engine, region: str, days: int = 7, historical_days: int = 30) -> Dict:
    """
    Analyze data quality for a specific region.
    
    Args:
        engine: SQLAlchemy engine
        region: Region code
        days: Number of days of recent data to analyze
        historical_days: Number of days of historical data to use for context
        
    Returns:
        Dictionary with data quality analysis
    """
    # Get recent data for analysis
    df = get_energy_data(engine, region, days)
    if df.empty:
        logger.warning(f"No data available for region {region}")
        return {
            'region': region,
            'score': 0,
            'checks': {},
            'data_available': False
        }
    
    # Get historical data for context
    historical_df = None
    if historical_days > days:
        historical_df = get_energy_data(engine, region, historical_days)
    
    # List of columns to check (excluding timestamp and region)
    sources = [col for col in df.columns if col in ALL_SOURCES]
    
    # Run all checks
    check_results = {
        'missing_data': check_missing_data(df, sources),
        'flatline': check_flatline(df, sources),
        'negative_values': check_negative_values(df, sources),
        'outliers': check_outliers(df, sources),
        'zero_values': check_zero_values(df, sources, historical_df)
    }
    
    # Calculate overall score
    overall_score = calculate_overall_quality_score(check_results)
    
    return {
        'region': region,
        'score': overall_score,
        'checks': check_results,
        'data_available': True,
        'timestamp': datetime.now().isoformat()
    }

def format_json_output(results: List[Dict]) -> str:
    """
    Format results as a JSON string.
    
    Args:
        results: List of results for each region
        
    Returns:
        Formatted JSON string
    """
    # Add summary statistics
    regions_with_data = [r for r in results if r['data_available']]
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {}
    }
    
    if regions_with_data:
        avg_score = sum(r['score'] for r in regions_with_data) / len(regions_with_data)
        best_region = max(regions_with_data, key=lambda r: r['score'])
        worst_region = min(regions_with_data, key=lambda r: r['score'])
        
        output['summary'] = {
            'average_score': round(avg_score, 1),
            'regions_analyzed': len(regions_with_data),
            'best_region': {
                'name': best_region['region'],
                'score': best_region['score']
            },
            'worst_region': {
                'name': worst_region['region'],
                'score': worst_region['score']
            }
        }
    
    # Convert to JSON string
    return json.dumps(output, indent=2)

def check_for_alerts(result: Dict) -> List[Dict]:
    """
    Check for alert conditions in the results.
    
    Args:
        result: Analysis result for a region
        
    Returns:
        List of alert dictionaries
    """
    if not result['data_available']:
        return [{
            'severity': 'high',
            'type': 'data_availability',
            'message': f"No data available for region {result['region']}",
            'region': result['region']
        }]
    
    alerts = []
    
    # Check overall score
    if result['score'] < ALERT_THRESHOLDS['overall_score']:
        alerts.append({
            'severity': 'high',
            'type': 'overall_quality',
            'message': f"Overall quality score for {result['region']} is below threshold: {result['score']}",
            'region': result['region'],
            'score': result['score']
        })
    
    # Check individual metrics
    for check_type, check_result in result['checks'].items():
        if check_type in ALERT_THRESHOLDS and check_result['overall'] < ALERT_THRESHOLDS[check_type]:
            alerts.append({
                'severity': 'medium',
                'type': check_type,
                'message': f"{check_type.replace('_', ' ').title()} score for {result['region']} is below threshold: {check_result['overall']:.1f}",
                'region': result['region'],
                'score': check_result['overall']
            })
            
            # Add specific details for critical sources if applicable
            if check_type == 'zero_values':
                for source in CRITICAL_SOURCES:
                    if source in check_result['details'] and check_result['details'][source]['score'] < 5:
                        alerts.append({
                            'severity': 'high',
                            'type': f'critical_source_{source}',
                            'message': f"Critical source {source} has quality issues in region {result['region']}",
                            'region': result['region'],
                            'source': source,
                            'zero_pct': check_result['details'][source]['zero_pct']
                        })
    
    return alerts

# AWS Integration Functions

def send_sns_alert(alert: Dict, sns_topic_arn: str = None):
    """
    Send an alert via AWS SNS (Simple Notification Service).
    
    Args:
        alert: Alert dictionary
        sns_topic_arn: ARN of the SNS topic to publish to
    """
    try:
        # This is a placeholder function - in a real implementation, use boto3
        logger.info(f"Would send SNS alert: {alert}")
        logger.info(f"To SNS topic: {sns_topic_arn or 'TOPIC_ARN_NOT_PROVIDED'}")
        
        # Actual implementation would look like:
        """
        import boto3
        
        sns_client = boto3.client('sns')
        
        message = f"ALERT: {alert['message']}"
        subject = f"Carbon Intensity Data Quality Alert: {alert['severity'].upper()} - {alert['type']}"
        
        response = sns_client.publish(
            TopicArn=sns_topic_arn,
            Message=json.dumps({
                'default': message,
                'email': message,
                'sms': f"{alert['severity'].upper()}: {alert['message']}"
            }),
            Subject=subject,
            MessageStructure='json'
        )
        """
    except Exception as e:
        logger.error(f"Error sending SNS alert: {e}")

def send_ses_email_report(results: Dict, recipients: List[str] = None):
    """
    Send a detailed email report via AWS SES (Simple Email Service).
    
    Args:
        results: Quality analysis results
        recipients: List of email recipients
    """
    try:
        # This is a placeholder function - in a real implementation, use boto3
        logger.info(f"Would send SES email report to: {recipients or 'RECIPIENTS_NOT_PROVIDED'}")
        
        # Actual implementation would look like:
        """
        import boto3
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        
        ses_client = boto3.client('ses')
        
        # Create a formatted HTML report
        html_body = "<html><body>"
        html_body += "<h1>Carbon Intensity Data Quality Report</h1>"
        html_body += f"<p>Report generated at: {results['timestamp']}</p>"
        
        html_body += "<h2>Summary</h2>"
        html_body += f"<p>Average Quality Score: {results['summary']['average_score']}/10</p>"
        html_body += f"<p>Regions Analyzed: {results['summary']['regions_analyzed']}</p>"
        html_body += f"<p>Best Region: {results['summary']['best_region']['name']} ({results['summary']['best_region']['score']}/10)</p>"
        html_body += f"<p>Worst Region: {results['summary']['worst_region']['name']} ({results['summary']['worst_region']['score']}/10)</p>"
        
        html_body += "<h2>Region Details</h2>"
        
        for region_result in results['results']:
            if region_result['data_available']:
                html_body += f"<h3>Region: {region_result['region']}</h3>"
                html_body += f"<p>Quality Score: {region_result['score']}/10</p>"
                
                html_body += "<table border='1'>"
                html_body += "<tr><th>Check Type</th><th>Score</th></tr>"
                
                for check_type, check_result in region_result['checks'].items():
                    html_body += f"<tr><td>{check_type.replace('_', ' ').title()}</td><td>{check_result['overall']:.1f}/10</td></tr>"
                
                html_body += "</table>"
            else:
                html_body += f"<h3>Region: {region_result['region']} - No data available</h3>"
        
        html_body += "</body></html>"
        
        message = MIMEMultipart()
        message['Subject'] = 'Carbon Intensity Data Quality Report'
        message['From'] = 'no-reply@your-domain.com'
        message['To'] = ', '.join(recipients)
        
        part = MIMEText(html_body, 'html')
        message.attach(part)
        
        response = ses_client.send_raw_email(
            Source='no-reply@your-domain.com',
            Destinations=recipients,
            RawMessage={'Data': message.as_string()}
        )
        """
    except Exception as e:
        logger.error(f"Error sending SES email report: {e}")

def main():
    parser = argparse.ArgumentParser(description='Data Quality Monitoring for Energy Metrics')
    parser.add_argument('--region', help='Region code to analyze (if not specified, all regions will be analyzed)')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to analyze')
    parser.add_argument('--historical-days', type=int, default=30, help='Number of days of historical data to use for context')
    parser.add_argument('--db-url', help='Database URL (optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--output-file', help='File to write JSON output to (optional)')
    parser.add_argument('--sns-topic', help='ARN of the SNS topic for alerts (optional)')
    parser.add_argument('--email-recipients', help='Comma-separated list of email recipients for reports (optional)')
    parser.add_argument('--alert', action='store_true', help='Send alerts for quality issues')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Connect to database
    engine = connect_to_database(args.db_url)
    
    # Get regions to analyze
    regions = [args.region] if args.region else get_regions(engine)
    
    if not regions:
        logger.error("No regions found for analysis")
        return
    
    # Analyze each region
    results = []
    all_alerts = []
    
    for region in regions:
        logger.info(f"Analyzing data quality for region {region}")
        result = analyze_region_data_quality(engine, region, args.days, args.historical_days)
        results.append(result)
        
        # Check for alerts
        if args.alert:
            region_alerts = check_for_alerts(result)
            all_alerts.extend(region_alerts)
            
            # Send SNS alerts if configured
            if args.sns_topic and region_alerts:
                for alert in region_alerts:
                    send_sns_alert(alert, args.sns_topic)
        
        # Print summary if not in JSON mode
        if not args.json:
            if result['data_available']:
                print(f"Region: {region}")
                print(f"Quality Score: {result['score']}/10")
                
                for check_type, check_result in result['checks'].items():
                    print(f"  {check_type.replace('_', ' ').title()}: {check_result['overall']:.1f}/10")
                
                print()
            else:
                print(f"Region: {region} - No data available\n")
    
    # Format results as JSON
    json_output = format_json_output(results)
    
    # Output JSON if requested
    if args.json:
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(json_output)
            logger.info(f"Results written to {args.output_file}")
        else:
            print(json_output)
    
    # Send email report if recipients are specified
    if args.email_recipients:
        recipients = [email.strip() for email in args.email_recipients.split(',')]
        parsed_json = json.loads(json_output)
        send_ses_email_report(parsed_json, recipients)
    
    # Print overall stats if not in JSON mode
    if not args.json:
        regions_with_data = [r for r in results if r['data_available']]
        if regions_with_data:
            avg_score = sum(r['score'] for r in regions_with_data) / len(regions_with_data)
            print(f"Average Quality Score: {avg_score:.1f}/10")
            
            # Find best and worst regions
            best_region = max(regions_with_data, key=lambda r: r['score'])
            worst_region = min(regions_with_data, key=lambda r: r['score'])
            
            print(f"Best Quality: {best_region['region']} ({best_region['score']}/10)")
            print(f"Worst Quality: {worst_region['region']} ({worst_region['score']}/10)")
            
        # Print alert summary
        if args.alert and all_alerts:
            print("\nAlerts:")
            for alert in all_alerts:
                print(f"  [{alert['severity'].upper()}] {alert['message']}")

if __name__ == "__main__":
    main() 