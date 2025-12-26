#!/bin/bash
# Quick Start Script for Research Validation
# Run this to generate publication-ready results

echo "=================================="
echo "RESEARCH VALIDATION QUICK START"
echo "=================================="

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

# Step 2: Run validation experiments
echo ""
echo "Step 2: Running validation experiments (100 episodes)..."
echo "This will take approximately 5-10 minutes..."
python validation_framework.py --episodes 100 --output results/

# Step 3: Display results
echo ""
echo "=================================="
echo "VALIDATION COMPLETE!"
echo "=================================="
echo ""
echo "Results saved to: results/"
echo ""
echo "Generated files:"
echo "  ✓ results/statistical_report.md"
echo "  ✓ results/comparison_boxplots.png"
echo "  ✓ results/learning_curve.png"
echo "  ✓ results/comparison_data.md"
echo "  ✓ results/rl_results.csv"
echo "  ✓ results/baseline_results.csv"
echo ""
echo "Next steps:"
echo "  1. Review results/statistical_report.md"
echo "  2. Include plots in your paper"
echo "  3. Use data from comparison_data.md for tables"
echo ""
echo "=================================="
