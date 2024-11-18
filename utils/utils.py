PAD_VAL = -1e9
P1, P2 = 0, 1
PRISONERS_E_DESCRIPTION = 'Repeated, two-player game where players can cooperate or defect'
CHICKEN_E_DESCRIPTION = 'Repeated, two-player game where players can swerve or go straight; if they both go ' \
                        'straight, they get damage; if one goes straight while the other sweres, the one that went ' \
                        'straight wins'
COORD_E_DESCRIPTION = 'Repeated, two-player game where players have two choices; if they both choose the same ' \
                      'action, they both win; if they choose different actions, they both lose'
RG_G_DESCRIPTIONS = {
    'AlgaaterCoop': 'Generator that tries to cooperate with the other player',
    'AlgaaterCoopPunish': 'Generator that tries to cooperate with the other player and attacks them if they do not cooperate in return',
    'AlgaaterBully': 'Generator that tries to take advantage of the other player',
    'AlgaaterBullyPunish': 'Generator that tries to take advantage of the other player and attacks them if there is any resistance',
    'AlgaaterBullied': 'Generator that allows other players to take advantage of it',
    'AlgaaterMinimax': 'Generator that tries to achieve the best outcome under the worst possible conditions',
    'AlgaaterCfr': 'Generator that tries to minimize regret'
}
FOREX_E_DESCRIPTION = 'Foreign exchange market trading, which can be viewed as a repeated, zero-sum game'
NETWORK_NAME = 'AATention'
FOREX_G_DESCRIPTIONS = {
    'BarMovement': 'Generator that uses recent price bars to look for big movements in a particular direction',
    'BeepBoop': 'Generator that uses histograms',
    'BollingerBands': 'Generator that looks for price breaking out of bands',
    'Choc': 'Generator that tries to look for movement beyond a key level',
    'KeltnerChannels': 'Generator that is similar to bollinger bands (uses bands), but the band values are calculated differently',
    'MACrossover': 'Generator that tracks two moving averages: one with a short period and another with a longer period',
    'MACD': 'Generator that looks for crossover on the MACD indicator',
    'MACDKeyLevel': 'Generator that looks for crossover on the MACD indicator, but also checks if price is near a key level',
    'MACDStochastic': 'Generator that looks for crossover on the MACD indicator, but also looks for stochastic crossovers',
    'PSAR': 'Generator that looks if there is a shift in PSAR values',
    'RSI': 'Generator that looks for overbought or oversold conditions',
    'SqueezePro': 'Generator that looks for price breaking out of a sideways pattern',
    'Stochastic': 'Generator that looks for stochastic crossover',
    'Supertrend': 'Generator that looks for shifts in supertrend lines'
}
HARE_NAME = 'hare'
STAG_NAME = 'stag'
AVAILABLE = -1  # A cell in the grid with this value means it is available/not taken
VERTICAL, HORIZONTAL = 'vertical', 'horizontal'
POSSIBLE_MOVEMENTS = [VERTICAL, HORIZONTAL]  # Up/down, left/right
POSSIBLE_DELTA_VALS = [-1, 0, 1]  # Left/down, stay, right/up
MAX_MOVEMENT_UNITS = 1
LEFT, RIGHT, UP, DOWN, NONE = 0, 1, 2, 3, 4
HARE_REWARD = 10
STAG_REWARD = 60
N_REQUIRED_TO_CAPTURE_HARE = 1
N_REQUIRED_TO_CAPTURE_STAG = 3
N_HUNTERS = 3
STAG_HARE_E_DESCRIPTION = 'A two-dimensional grid environment where three hunters can hunt a hare or work together to ' \
                          'hunt a stag'
STAG_HARE_G_DESCRIPTIONS = [
    'Generator that greedily pursues the hare',
    'Generator that greedily pursues the hare but tries to plan against collisions',
    'Generator that greedily pursues the stag but tries to plan against collisions',
    'Generator that intelligently pursues the stag and tries to plan against collisions and help its team members'
]

