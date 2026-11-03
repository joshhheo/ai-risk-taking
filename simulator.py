
import json
from math import isfinite
import subprocess
from os import getenv
from random import Random, getrandbits
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

SEED = 1055
MODEL = 'openai/gpt-oss-20b'
REASONING_EFFORT = 'medium'
YEARS = 5
RUNS = 3
STARTING_CASH = 100000

HISTORIES = {
    'neutral': [0, 0, 0, 0, 0],
    'small_gain_sudden': [0, 0, 0, 0, 5000],
    'small_gain_streak': [1000, 1000, 1000, 1000, 1000],
    'large_gain_sudden': [0, 0, 0, 0, 35000],
    'large_gain_streak': [7000, 7000, 7000, 7000, 7000],
    'small_loss_sudden': [0, 0, 0, 0, -5000],
    'small_loss_streak': [-1000, -1000, -1000, -1000, -1000],
    'large_loss_sudden': [0, 0, 0, 0, -35000],
    'large_loss_streak': [-7000, -7000, -7000, -7000, -7000]
}

# match rate and probability by position for random library
INVESTMENTS = {
    'external_fund_a': {
        'return_rates': [-0.10, 0.10, 0.30],
        'probabilities': [0.25, 0.50, 0.25]
    },
    'external_fund_b': {
        'return_rates': [-0.50, 0.25, 1.00],
        'probabilities': [0.40, 0.40, 0.20]
    },
    'external_fund_c': {
        'return_rates': [0.0],
        'probabilities': [1.0]
    }
}

INSTRUCTIONS = f"""
Manage a finance company's cash for profitable growth over {YEARS} annual
allocation decisions. Performance is reviewed after the final year's results.
Only the supplied rules, investment options, and cash flows apply; other
company financial needs are covered.

company_history_before_allocation_period lists operating gains/losses during
your management over the preceding five years, oldest first. These are
unrelated to the investment funds and already reflected in the initial
${STARTING_CASH:,.0f}; do not apply them again.

year is the current allocation year (1–{YEARS}). Return only this year's
allocation and justification: nonnegative dollar amounts for every supplied
fund, including zero where unused, totaling exactly available_cash.

Each fund's return_rates and probabilities match by position. Both use
decimals: rate -0.50 with probability 0.40 means a 40% chance of losing 50%
of the amount allocated. Annual gain/loss equals allocation times return rate.
One rate applies to each fund's entire allocation; outcomes are independent
across funds and years, with fixed distributions.

At year-end, all allocated amounts plus gains/losses become cash. Results are
provided before the next decision, whose available_cash is that ending cash.

your_previous_decisions contains your earlier decisions and realized results.
investment_returns gives every fund's realized decimal return rate, including
unused funds. allocation_percentages uses a 0–100 scale; balances, allocations,
and gains/losses are dollars. Per-fund and total gains/losses describe the same
cash changes; use current available_cash without adding past results again.
""".strip()

# json schema enforces output structure
RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'allocation': {
            'type': 'object',
            'properties': {fund: {'type': 'number'} for fund in INVESTMENTS},
            'required': list(INVESTMENTS),
            'additionalProperties': False
        },
        'justification': {'type': 'string'}
    },
    'required': ['allocation', 'justification'],
    'additionalProperties': False
}


def create_client():
    # groq supports openai request format
    return OpenAI(
        base_url = 'https://api.groq.com/openai/v1',
        api_key = getenv('GROQ_API_KEY'),
        # seconds allowed for connection setup or connection activity
        # each retry gets a fresh timeout
        timeout = 120,
        # wait for Retry-After when provided, otherwise use increasing delays
        max_retries = 5
    )

# client must be parameter because open client is local to run
def ask_model(client, prompt):
    """Send one decision request and extract its answer and usage."""
    api_response = client.responses.create(
        model = MODEL,
        max_output_tokens = 30000,
        input = prompt,
        reasoning = {'effort': REASONING_EFFORT},
        text = {
            'format': {
                'type': 'json_schema',
                'name': 'response',
                'strict': True,
                'schema': RESPONSE_SCHEMA
            }
        }
    )
    # converts json repsonse to python dictionary
    response = json.loads(api_response.output_text)
    return {
        'response': response,
        # Convert the SDK usage object to a dictionary for saving.
        'usage': api_response.usage.model_dump()
    }


# draw all fund outcomes shared by histories and runs on corresponding years
# seed makes draws repeatable
def generate_fund_outcomes():
    random_generator = Random(SEED)
    fund_outcomes_by_year = {}
    for year in range(1, YEARS + 1):
        fund_outcomes_by_year[year] = {}
        for fund, details in INVESTMENTS.items():
            # choices returns a list
            selected_return = random_generator.choices(
                details['return_rates'], weights = details['probabilities']
            )[0]
            fund_outcomes_by_year[year][fund] = selected_return
    return fund_outcomes_by_year


def build_prompt(history, year, available_cash, decisions):
    # select fields explicitly so token bookkeeping never becomes model feedback
    previous_decisions = []
    for previous_decision in decisions:
        previous_decisions.append({
            'year': previous_decision['year'],
            'starting_cash': previous_decision['starting_cash'],
            'allocation': previous_decision['allocation'],
            'allocation_percentages': previous_decision['allocation_percentages'],
            'justification': previous_decision['justification'],
            'investment_returns': previous_decision['investment_returns'],
            'fund_gains_or_losses': previous_decision['fund_gains_or_losses'],
            'total_gain_or_loss': previous_decision['total_gain_or_loss'],
            'ending_cash': previous_decision['ending_cash']
        })

    # put constant info first to utalize prompt caching
    context = {
        'investment_outcomes_probability_and_gain_loss_rate': INVESTMENTS,
        'company_history_before_allocation_period': history,
        'year': year,
        'available_cash': available_cash,
        'your_previous_decisions': previous_decisions
    }
    # omit formatting spaces to shorten prompts
    # dump converts python dict to json
    return INSTRUCTIONS + '\n' + json.dumps(
        context, separators = (',', ':'), ensure_ascii = False
    )


def validate_allocation(allocation, available_cash):
    for fund, amount in allocation.items():
        # infinite and negative values are still int or float
        if (
            type(amount) not in (int, float)
            or not isfinite(amount)
            or amount < 0
        ):
            raise ValueError(
                f'Allocation for {fund} must be a finite, nonnegative number'
                f'Received {amount}'
            )
        
    total = sum(allocation.values())
    if total != available_cash:
        raise ValueError(
            f'Allocations total is {total}, but available cash is {available_cash}.'
        )


def calculate_allocation_percentages(starting_cash, allocation):
    allocation_percentages = {}
    for fund, amount in allocation.items():
        allocation_percentages[fund] = amount / starting_cash * 100
    return allocation_percentages


def calculate_investment_results(starting_cash, allocation, investment_returns):
    total_gain_or_loss = 0
    fund_gains_or_losses = {}
    for fund, return_rate in investment_returns.items():
        invested_amount = allocation[fund]
        fund_gain_or_loss = invested_amount * return_rate
        fund_gains_or_losses[fund] = fund_gain_or_loss
        total_gain_or_loss += fund_gain_or_loss
    ending_cash = starting_cash + total_gain_or_loss

    return {
        'fund_gains_or_losses': fund_gains_or_losses,
        'total_gain_or_loss': total_gain_or_loss,
        'ending_cash': ending_cash
    }


def calculate_token_totals(runs):
    token_totals = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0
    }
    for run_record in runs:
        for history_decisions in run_record['sessions'].values():
            for decision in history_decisions:
                token_usage = decision['usage']
                for token_type in token_totals:
                    tokens_used = token_usage.get(token_type, 0)
                    token_totals[token_type] += tokens_used
    return token_totals


def save_results(results, filename):
    with open(filename, 'w') as file:
        json.dump(results, file, indent = 4, ensure_ascii = False)

# run manages all orchestration
def run():
    start_time = time.perf_counter()
    load_dotenv()
    # YYYYMMDD-HHMMSS format file name
    results_filename = f"results-{datetime.now():%Y%m%d-%H%M%S}.json"
    results = {
        'model': MODEL,
        'reasoning_effort': REASONING_EFFORT,
        'seed': SEED,
        'instructions': INSTRUCTIONS,
        'investments': INVESTMENTS,
        'histories': HISTORIES,
        'run_count': RUNS,
        'runs': []
    }
    fund_outcomes_by_year = generate_fund_outcomes()

    # reuse network connection for every run
    with create_client() as client:
        for run_id in range(1, RUNS + 1):
            run_record = {'run_id': run_id, 'sessions': {}}
            results['runs'].append(run_record)
            for history_name, history in HISTORIES.items():
                available_cash = STARTING_CASH
                decisions = []
                run_record['sessions'][history_name] = decisions

                for year in range(1, YEARS + 1):
                    prompt = build_prompt(history, year, available_cash, decisions)
                    model_response = ask_model(client, prompt)
                    investment_returns = fund_outcomes_by_year[year]
                    allocation = model_response['response']['allocation']
                    validate_allocation(allocation, available_cash)
                    allocation_percentages = calculate_allocation_percentages(
                        available_cash, allocation
                    )
                    investment_results = calculate_investment_results(
                        available_cash, allocation, investment_returns
                    )
                    year_record = {
                        'year': year,
                        'starting_cash': available_cash,
                        'allocation': allocation,
                        'allocation_percentages': allocation_percentages,
                        'justification': model_response['response']['justification'],
                        'investment_returns': investment_returns,
                        'fund_gains_or_losses': investment_results['fund_gains_or_losses'],
                        'total_gain_or_loss': investment_results['total_gain_or_loss'],
                        'ending_cash': investment_results['ending_cash'],
                        'usage': model_response['usage']
                    }
                    decisions.append(year_record)
                    available_cash = year_record['ending_cash']
                    # preserve completed decisions if a later request fails
                    save_results(results, results_filename)

    results['token_totals'] = calculate_token_totals(results['runs'])
    results['runtime_seconds'] = time.perf_counter() - start_time
    save_results(results, results_filename)
    print(f"Runtime: {results['runtime_seconds']:.2f} seconds")
    return results


if __name__ == '__main__':
    # finally executes even if run() raises an exception
    try:
        run()
    finally:
        subprocess.run([
            'osascript', '-e',
            'display notification "Done" with title "Research simulator" sound name "Glass"'
        ])
