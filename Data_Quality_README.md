# Carbon Aggregation Service (CAS) Data Quality Monitor

Goal is to analyze the energy generation (production) data we collect across different regions and checks for various quality issues. This will help us maintain high data integrity before it hits our forecasting models.

## What data quality checks are performed?


- **Multiple Quality Checks**: Catches all sorts of issues:
  - Missing data (those pesky NULLs)
  - Flatlined values (when nothing changes for suspiciously long periods)
  - Negative values (physically impossible for generation!)
  - Statistical outliers (those "wait, what?" data points)
  - Suspicious zero values (with special logic for different energy types)

- **Smart Energy Source Analysis**:
  - Special handling for intermittent sources (SUN, WND) that can naturally be zero
  - Higher penalties when critical sources (COL, NUC, NG) zero out (that probably means data issues)
  - Uses historical patterns to learn what's "normal" for each region

- **Simple Scoring System**:
  - Each check gives a 0-10 score (10 is perfect)
  - Combines into an overall quality score
  - Makes it easy to set up alerts and monitoring thresholds

- **SmartThing Cloud Integration**:
  - Built to integrate with our AWS environment
  - JSON outputs ready for our API endpoints
  - Works with our existing database structure

## Setting Up

You will need to integrate this with our AWS environment, lambda function and datadog (check with Prathmesh or Prashant for the details).

- Python 3.6+ (I used 3.9 during development)

```bash
pip install pandas numpy sqlalchemy
```

For our AWS notification-service integration:

```bash
pip install boto3
```

## Configuration

I've used the development settings I had shared in the base code for the aggregation service , but you can tweak these based on the latest DB changes you all have made.

- **Database Connection**: Default is `sqlite:///data/energy_metrics.db` but you can point it to our AWS RDS if needed (Prashant has the connection strings)
- **Alert Thresholds**: To discuss with @Prathmesh (Sreeni/Kaushik) --- Adjust the `ALERT_THRESHOLDS` dictionary if you want to make alerts more/less sensitive
- **Quality Check Weights**: Change how different checks influence the overall score in `DEFAULT_WEIGHTS`

## How to Use It

### Basic Check

```bash
python data_quality_monitor.py
```

This will scan all regions for the last 7 days, with 30 days of history for context.

### Check a Specific Region

```bash
python data_quality_monitor.py --region caiso
```

Good for troubleshooting specific regions that are acting up.

### Longer Analysis Period

```bash
python data_quality_monitor.py --days 14 --historical-days 90
```

Useful for catching issues that develop over longer periods.

### Get JSON Output

```bash
python data_quality_monitor.py --json
```

I am not sure about the whole architecture of the API integrations - but i belive this will work along with the datadog integration also. 

### Save Results to a File

```bash
python data_quality_monitor.py --json --output-file quality_report.json
```

Helpful for manual reviews or sharing results with the team.

### Trigger Alerts

```bash
python data_quality_monitor.py --alert
```

This will send alerts based on our thresholds.

### Email Reports to the Team

```bash
python data_quality_monitor.py --email-recipients "sayonsom.c@smartthing.com,prathmesh.m@smartthing.com"
```

Weekly reports might be good for this - I'll set up a cron job.

### Complete AWS Integration Example

```bash
python data_quality_monitor.py --region caiso --days 7 --json --alert --sns-topic arn:aws:sns:us-east-1:123456789012:smartthing-data-quality-alerts --email-recipients "team@smartthing.com" --verbose
```

## Command Line Options 

| Option | What it does |
|--------|-------------|
| `--region` | Specify a region to analyze (omit to check all regions) |
| `--days` | Days of recent data to analyze (default: 7) |
| `--historical-days` | Days of historical data for context (default: 30) |
| `--db-url` | Override the database URL |
| `--verbose` | Show detailed logs (helpful for debugging) |
| `--json` | Output in JSON format (for APIs and Datadog) |
| `--output-file` | Save JSON to a file |
| `--sns-topic` | SNS topic ARN for alerts (get this from Prathmesh) |
| `--email-recipients` | Who should get the email reports |
| `--alert` | Enable alerting |

## JSON Output Format

The JSON looks like this (simplified):

```json
{
  "timestamp": "2023-06-15T14:32:45.123456",
  "results": [
    {
      "region": "caiso",
      "score": 8,
      "checks": {
        "missing_data": { "overall": 9.8, "details": {...} },
        "flatline": {...},
        "negative_values": {...},
        "outliers": {...},
        "zero_values": {...}
      },
      "data_available": true,
      "timestamp": "2023-06-15T14:32:44.123456"
    },
    ...
  ],
  "summary": {
    "average_score": 7.5,
    "regions_analyzed": 3,
    "best_region": { "name": "caiso", "score": 8 },
    "worst_region": { "name": "ercot", "score": 6 }
  }
}
```

Check with @Prathmesh (PrashantSreeni/Kaushik) for datadog integration.

## AWS Integration Details

### SNS Alerts

I've set up the script to send alerts to AWS SNS when data quality drops:

- No data for a region
- Overall quality score too low
- Individual checks failing
- Critical energy sources with suspicious patterns

To use this with our SmartThing Cloud:

1. Please create and update the SNS topic ARN from our AWS environment (Prathmesh should have this)
2. Need to make sure my AWS credentials have SNS publish permissions
3. Run with the `--alert --sns-topic YOUR_TOPIC_ARN` flags

Prathmesh mentioned we might want different SNS topics for different severity levels - let me know if we should split these up.

### SES Email Reports

For scheduled email reports:

1. I'll need our verified SES identities (check with Prashant)
2. Make sure my role has SES send permissions
3. Run with `--email-recipients "sayonsom.c@samsung.com,prathmesh.m@samsung.com"`

Default is a weekly email to the team with summary stats, and daily emails if scores drop below 6.

## Understanding the Scores

When reviewing reports:

### Overall Score
- **9-10**: We're good! Data looks clean
- **7-8**: Minor issues, but usable
- **5-6**: We should investigate
- **3-4**: Definite problems - forecasts may be affected
- **0-2**: Major issues - don't use this data!

The individual checks help pinpoint what exactly is wrong.

## Extending This

### Adding New Checks

If you want to add a new check:

1. Create a function similar to the existing check functions
2. Add it to `check_results` in `analyze_region_data_quality()`
3. Add a weight in `calculate_overall_quality_score()`
4. Add a threshold in `ALERT_THRESHOLDS`



### Runtime Issues
- Use `--verbose` for detailed logs
- For slow processing, use a smaller `--days` value
- Check the individual scores to find the specific problem


## Topics to discuss with Team


- SNS topic ARNs for different alert levels
- SES email verification status
- IAM permissions for the scheduled job
- Does this need to integrate with Datadog?
- Team email distribution lists
- AWS RDS connection details .. will data quality need a separate DB?
