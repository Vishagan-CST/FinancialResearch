import sys
sys.path.insert(0, 'backend')
import warnings
warnings.filterwarnings('ignore')
from inference_pipeline import InferencePipeline

p = InferencePipeline()

article = """
Germany's economy expands by 0.3% in the second quarter of 2026, slightly exceeding the preliminary estimate of 0.2%.
The revised figure strengthened evidence that Europe's largest economy may be emerging from a prolonged period of weak growth.
Reuters reported that the improvement was supported by stronger business activity, improving export orders,
and better-than-expected investor sentiment.

Several indicators have pointed toward a gradual recovery. Investor confidence rose more than economists had expected
earlier in August, reflecting optimism about corporate earnings and international demand. Export-oriented industries
benefited from an increase in overseas orders, helping manufacturing regain momentum after a difficult period marked
by high energy costs and global economic uncertainty.

Business surveys also showed encouraging results. Germany's manufacturing sector recorded its strongest expansion in
more than four years, helping offset weakness in some service-sector activities. Employment conditions stabilized,
and companies reported improved expectations for future business conditions. Analysts noted that the manufacturing
rebound is becoming an increasingly important driver of economic growth.

The positive developments are not limited to Germany. Across the euro zone, business activity has reached its fastest
pace since late 2025. New orders have increased significantly, export demand has improved, and employment growth has
resumed in several sectors. These broader European trends provide additional support for Germany's recovery because of
the country's strong dependence on trade with neighboring economies.

Economists caution that challenges remain, including geopolitical tensions, energy-price volatility, and inflationary
pressures. Nevertheless, the latest data suggest that the German economy has demonstrated greater resilience than many
analysts expected earlier in the year. Some financial institutions have already revised their growth forecasts upward,
citing stronger exports, improving business confidence, and government efforts aimed at boosting competitiveness and
investment.
"""

result = p.predict(article)

print("=== PREDICTION ===")
print(f"  Result    : {result['prediction'].upper()}")
print(f"  UP prob   : {result['probabilities']['up']:.4f}")
print(f"  DOWN prob : {result['probabilities']['down']:.4f}")
print()
print("=== FEATURES EXTRACTED ===")
for k, v in result['features'].items():
    print(f"  {k:<40} : {v:.4f}")
