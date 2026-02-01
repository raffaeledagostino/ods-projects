# Semi-Supervised Learning with Optimization Algorithms

Graph-based semi-supervised learning for binary classification with minimal labeled data.

## Overview

**Problem:** Minimize graph Laplacian loss with only 10% labeled data  
**Algorithms:** Gradient Descent, Block Coordinate GD (Gauss-Southwell), Coordinate Minimization

## Key Results

### Synthetic Dataset (1000 points, 10% labeled)
- All methods: **95.22% accuracy**
- Convergence: CM (52 iter) > GD (109 iter) >> BCGD (78,754 iter)

### Banknote Dataset (1372 points, 4 features)
- **GD/CM: 98.95%** | BCGD: 97.33%

## Technical Details

- **Similarity:** Gaussian kernel `exp(-k||x_i - x_j||²)`
- **Step-size:** `1/L` (L = max eigenvalue of Hessian)
- **Best method:** Coordinate Minimization (fastest, closed-form updates)

# Matrix Completion for Recommender Systems

Projection-free optimization for low-rank matrix recovery in collaborative filtering.

## Overview

**Problem:** `min ||X - U||² s.t. ||X||_* ≤ τ` (nuclear norm constraint)  
**Algorithms:** Frank-Wolfe (FW), Pairwise Frank-Wolfe (PFW), Projected Gradient (PG)

## Key Results

| Dataset | Best Method | RMSE | Rank | Time |
|---------|------------|------|------|------|
| Amazon Gift Cards (377 users) | **FW** | 1.537 | 2 | 0.65s |
| Netflix 2005 (4695 users) | **PG** | 0.889 | 96 | 19.4s |
| MovieLens 100k (943 users) | **PG** | 0.951 | 115 | 260s |

## Technical Highlights

- **FW:** Projection-free via Linear Minimization Oracle (dominant SVD)
- **PFW:** Away-steps reduce active set size
- **PG:** Projection onto nuclear norm ball via SVD shrinkage
- **Step-size:** Exact line search for optimal convergence

## Key Findings

- **Small datasets:** FW fastest with low-rank solutions
- **Large datasets:** PG best accuracy but computationally expensive
- **PFW:** Slowest, produces high-rank matrices
