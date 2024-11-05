#%%
random_combinations = [
    (35, 0.61, 0.61, 'test'),
    (45, 0.31, 0.91, 'validation'),
    (25, 0.01, 0.61, 'test'),
    (40, 0.91, 0.61, 'test')
] * 75  # Should create 300 items

for i, (n_topics, alpha_value, beta_value, train_eval_type) in enumerate(random_combinations):
    print(f"Iteration {i+1} - Number of topics: {n_topics}, Alpha: {alpha_value}, Beta: {beta_value}, Type: {train_eval_type}")