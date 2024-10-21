PAD_VAL = -1e9
P1, P2 = 0, 1
PRISONERS_E_DESCRIPTION = 'Repeated, two-player game where players can cooperate or defect'
CHICKEN_E_DESCRIPTION = 'Repeated, two-player game where players can swerve or go straight; if they both go ' \
                        'straight, they get damage; if one goes straight while the other sweres, the one that went ' \
                        'straight wins'
COORD_E_DESCRIPTION = 'Repeated, two-player game where players have two choices; if they both choose the same ' \
                      'action, they both win; if they choose different actions, they both lose'
PRISONERS_G_DESCRIPTIONS = {
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
