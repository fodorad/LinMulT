from linear_mult.models.MulT import MulT

# OOB + WFR
model = MulT((25, 35, 512, 256, 1024), 5, weights='fi-linmult-oob-wfr') # MT
model = MulT((25, 35, 512, 256, 1024), 1, weights='fi-linmult-oob-wfr-0') # TW O
model = MulT((25, 35, 512, 256, 1024), 1, weights='fi-linmult-oob-wfr-1') # TW C
model = MulT((25, 35, 512, 256, 1024), 1, weights='fi-linmult-oob-wfr-2') # TW E
model = MulT((25, 35, 512, 256, 1024), 1, weights='fi-linmult-oob-wfr-3') # TW A
model = MulT((25, 35, 512, 256, 1024), 1, weights='fi-linmult-oob-wfr-4') # TW N

# OOB
model = MulT((25, 35, 768), 5, weights='fi-linmult-oob') # MT
model = MulT((25, 35, 768), 1, weights='fi-linmult-oob-0') # TW O
model = MulT((25, 35, 768), 1, weights='fi-linmult-oob-1') # TW C
model = MulT((25, 35, 768), 1, weights='fi-linmult-oob-2') # TW E
model = MulT((25, 35, 768), 1, weights='fi-linmult-oob-3') # TW A
model = MulT((25, 35, 768), 1, weights='fi-linmult-oob-4') # TW N

# OOB old
model = MulT((25, 35, 768), 1, projected_modality_dim=30, weights='fi-linmult-oob-0-old') # TW O
model = MulT((25, 35, 768), 1, projected_modality_dim=30, weights='fi-linmult-oob-1-old') # TW O
model = MulT((25, 35, 768), 1, projected_modality_dim=30, weights='fi-linmult-oob-2-old') # TW O
model = MulT((25, 35, 768), 1, projected_modality_dim=30, weights='fi-linmult-oob-3-old') # TW O
model = MulT((25, 35, 768), 1, projected_modality_dim=30, weights='fi-linmult-oob-4-old') # TW O