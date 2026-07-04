"""Compute Phy-Q scores per template from agent results CSV and compare to other agents."""
import pandas as pd
import numpy as np

# ============================================================================
# PART 1: PDDL Agent Performance by Template
# ============================================================================

# Load PDDL agent results
df = pd.read_csv('pddl_agent_results.csv')

# Create win indicator
df['win'] = (df['result'] == 'Win').astype(int)

print('='*80)
print('PHY-Q SCORES BY TEMPLATE - PDDL AGENT (Local Generalization)')
print('='*80)

# Per-template breakdown
print('\n## Per-Template Performance')
print('-'*80)
header = f"{'Template':^10} | {'Train Win%':^12} | {'Test Win%':^12} | {'Gen Gap':^10} | {'Train N':^8} | {'Test N':^8}"
print(header)
print('-'*80)

templates = sorted(df['template'].unique())
all_results = []

for template in templates:
    t_df = df[df['template'] == template]
    
    train_df = t_df[t_df['mode'] == 'train']
    test_df = t_df[t_df['mode'] == 'test']
    
    train_wins = train_df['win'].sum()
    train_total = len(train_df)
    train_pct = (train_wins / train_total * 100) if train_total > 0 else 0
    
    test_wins = test_df['win'].sum()
    test_total = len(test_df)
    test_pct = (test_wins / test_total * 100) if test_total > 0 else 0
    
    gen_gap = train_pct - test_pct
    
    all_results.append({
        'template': template,
        'train_pct': train_pct,
        'test_pct': test_pct,
        'gen_gap': gen_gap,
        'train_n': train_total,
        'test_n': test_total
    })
    
    print(f'{template:^10} | {train_pct:^12.1f}% | {test_pct:^12.1f}% | {gen_gap:^+10.1f}% | {train_total:^8} | {test_total:^8}')

print('-'*80)

# Overall statistics
total_train = df[df['mode'] == 'train']
total_test = df[df['mode'] == 'test']

overall_train_pct = total_train['win'].mean() * 100
overall_test_pct = total_test['win'].mean() * 100
overall_gen_gap = overall_train_pct - overall_test_pct

print(f"{'OVERALL':^10} | {overall_train_pct:^12.1f}% | {overall_test_pct:^12.1f}% | {overall_gen_gap:^+10.1f}% | {len(total_train):^8} | {len(total_test):^8}")
print('='*80)

# PhyQ Score calculation
overall_accuracy = df['win'].mean() * 100

print('\n## PHY-Q SCORE SUMMARY (PDDL Agent)')
print('='*80)
print(f'Overall Accuracy (all levels):     {overall_accuracy:.1f}%')
print(f'Train Performance:                 {overall_train_pct:.1f}% ({len(total_train)} levels)')
print(f'Test Performance (Generalization): {overall_test_pct:.1f}% ({len(total_test)} levels)')
print(f'Generalization Gap:                {overall_gen_gap:+.1f}%')

# ============================================================================
# PART 2: COMPARISON WITH OTHER AGENTS
# ============================================================================

print('\n')
print('='*80)
print('COMPARISON WITH OTHER AGENTS (Phy-Q Benchmark)')
print('='*80)

# Load comparison data
try:
    comp_df = pd.read_csv('agent_comparison_results.csv')
    
    # Get unique agents (excluding PDDLAgent which has Win/Lose format)
    other_agents = comp_df[comp_df['agent'] != 'PDDLAgent']['agent'].unique()
    
    # Baseline agents have pass rates (0.0-1.0) in the 'result' column
    # Compute pass rate per template for each agent
    
    baseline_by_template = {}
    baseline_overall = {}
    
    for agent in other_agents:
        agent_df = comp_df[comp_df['agent'] == agent]
        # Convert result to float (pass rate)
        agent_df = agent_df.copy()
        agent_df['pass_rate'] = pd.to_numeric(agent_df['result'], errors='coerce')
        
        # Per-template pass rate (average across variations within template)
        template_rates = {}
        for template in templates:
            t_data = agent_df[agent_df['template'] == template]['pass_rate']
            if len(t_data) > 0:
                template_rates[template] = t_data.mean() * 100
            else:
                template_rates[template] = np.nan
        
        baseline_by_template[agent] = template_rates
        
        # Overall pass rate
        overall_rate = agent_df['pass_rate'].mean() * 100
        baseline_overall[agent] = overall_rate
    
    # Print comparison table - Per Template
    print('\n## Pass Rate by Template (All Agents)')
    print('-'*100)
    
    # Header
    header_parts = ['Agent'.ljust(20)]
    for t in templates:
        header_parts.append(f'T{t}'.center(10))
    header_parts.append('Overall'.center(10))
    print(' | '.join(header_parts))
    print('-'*100)
    
    # PDDL Agent row (using our computed results)
    pddl_row = ['PDDL Agent'.ljust(20)]
    for r in all_results:
        combined = (r['train_pct'] * r['train_n'] + r['test_pct'] * r['test_n']) / (r['train_n'] + r['test_n'])
        pddl_row.append(f"{combined:.1f}%".center(10))
    pddl_row.append(f"{overall_accuracy:.1f}%".center(10))
    print(' | '.join(pddl_row))
    
    # Other agents
    for agent in sorted(other_agents):
        row = [agent[:20].ljust(20)]
        for template in templates:
            rate = baseline_by_template[agent].get(template, np.nan)
            if pd.notna(rate):
                row.append(f"{rate:.1f}%".center(10))
            else:
                row.append('N/A'.center(10))
        row.append(f"{baseline_overall[agent]:.1f}%".center(10))
        print(' | '.join(row))
    
    print('-'*100)
    
    # Ranking by overall performance
    print('\n## Overall Performance Ranking (single_force scenario)')
    print('-'*60)
    
    all_agents_overall = {'PDDL Agent': overall_accuracy}
    all_agents_overall.update(baseline_overall)
    
    # Sort by performance
    ranked = sorted(all_agents_overall.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':^6} | {'Agent':<25} | {'Pass Rate':^12}")
    print('-'*60)
    for i, (agent, rate) in enumerate(ranked, 1):
        marker = ' <-- Our Agent' if agent == 'PDDL Agent' else ''
        print(f"{i:^6} | {agent:<25} | {rate:^12.1f}%{marker}")
    
    print('-'*60)
    
    # Find PDDL agent rank
    pddl_rank = next(i for i, (a, _) in enumerate(ranked, 1) if a == 'PDDL Agent')
    print(f"\nPDDL Agent Rank: {pddl_rank}/{len(ranked)}")
    
    # Comparison with specific baselines
    print('\n## Comparison vs Key Baselines')
    print('-'*60)
    
    key_baselines = ['Human', 'Datalab', 'Eagle\'s Wing', 'Bambirds', 'Random Agent']
    for baseline in key_baselines:
        if baseline in baseline_overall:
            diff = overall_accuracy - baseline_overall[baseline]
            status = 'BETTER' if diff > 0 else ('SAME' if diff == 0 else 'WORSE')
            print(f"vs {baseline:<20}: {diff:+.1f}% ({status})")
    
    print('='*80)
    
    # Per-template comparison
    print('\n## Per-Template Comparison: PDDL Agent vs Human')
    print('-'*80)
    
    if 'Human' in baseline_by_template:
        print(f"{'Template':^10} | {'PDDL':^12} | {'Human':^12} | {'Diff':^12} | {'Status':^10}")
        print('-'*80)
        
        for r in all_results:
            t = r['template']
            pddl_combined = (r['train_pct'] * r['train_n'] + r['test_pct'] * r['test_n']) / (r['train_n'] + r['test_n'])
            human_rate = baseline_by_template['Human'].get(t, np.nan)
            
            if pd.notna(human_rate):
                diff = pddl_combined - human_rate
                status = 'BETTER' if diff > 0 else ('SAME' if diff == 0 else 'WORSE')
                print(f"{t:^10} | {pddl_combined:^12.1f}% | {human_rate:^12.1f}% | {diff:^+12.1f}% | {status:^10}")
            else:
                print(f"{t:^10} | {pddl_combined:^12.1f}% | {'N/A':^12} | {'N/A':^12} | {'N/A':^10}")
        
        print('-'*80)

except FileNotFoundError:
    print("Warning: agent_comparison_results.csv not found. Skipping comparison.")

# ============================================================================
# PART 3: EXPORT SUMMARY
# ============================================================================

print('\n')
print('='*80)
print('SUMMARY EXPORT')
print('='*80)

# Summary table for export
summary_data = []
for r in all_results:
    summary_data.append({
        'Template': r['template'],
        'Train Win%': f"{r['train_pct']:.1f}%",
        'Test Win%': f"{r['test_pct']:.1f}%", 
        'Gen Gap': f"{r['gen_gap']:+.1f}%",
        'Train N': r['train_n'],
        'Test N': r['test_n']
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary to CSV
summary_df_full = pd.DataFrame(all_results)
summary_df_full.to_csv('phyq_scores_by_template.csv', index=False)
print(f'\nSaved detailed scores to: phyq_scores_by_template.csv')
