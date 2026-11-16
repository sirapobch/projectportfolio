# LLM-Derived Sentiment for Magnificent Seven Stock Prediction

**Question.** Does adding news sentiment from FinBERT (a finance-specific encoder) or Qwen2.5-7B-Instruct
(a general-purpose LLM) to a price, fundamental and macroeconomic baseline improve out-of-sample prediction and
portfolio performance for the seven largest US technology stocks, and are any return differences statistically significant?

**Design.** Three feature sets (baseline, +FinBERT, +Qwen) are each used by three models (Ridge logistic regression,
XGBoost, LSTM). Data, features, validation, hyperparameters and portfolio rules are held fixed, so only the sentiment
input changes.

1. **Data:** prices (Yahoo Finance), macroeconomic series (FRED), point-in-time SEC XBRL fundamentals and 55,776 scored news headlines, 2010 to 2025
2. **Exploratory analysis** before any model is fitted
3. **Purged walk-forward validation** (2016 to 2025, refit every 126 days) and a weekly long-only top-3 portfolio (inverse-volatility weights, VIX scaling, drawdown breaker, 5 bp costs)
4. **Evaluation:** classification metrics, rank IC with Newey-West t-statistics, return and Sharpe ratio
5. **Significance tests** on return: paired block-bootstrap / block-permutation

## Files

```
mag7_sentiment_model_comparison.ipynb   Full pipeline (outputs cleared)
```

## Running it

1. Open the notebook in Google Colab (the original run used a T4 GPU).
2. Run Cell 0 to install dependencies. Market, macro and SEC data are downloaded and cached in `data_cache/`.
3. In Section 1, set `SEC_USER_AGENT` to `"Your Name your@email"` (SEC EDGAR rejects anonymous requests).
4. Provide `mag7_combined_dataset.csv`, the headline file with FinBERT and Qwen scores. It is not included in this repo.
5. `QUICK_VERIFY = True` runs a fast smoke test. Set it to `False` for the full run.
