from dataclasses import dataclass


@dataclass
class MarketSimulationResults:
    reward: float
    day_fees: float
    net_reward: float
    avg_pips_risked: float
    n_buys: int
    n_sells: int
    n_wins: int
    n_losses: int
    longest_win_streak: int
    longest_loss_streak: int
    starting_account_balance: float
    lowest_account_balance: float
    highest_account_balance: float
    account_balance: float
    _curr_win_streak: int
    _curr_loss_streak: int

    # Helper function to update the simulation results once a trade closes out
    def update_results(self, trade_amount: float, day_fees: float) -> None:
        self.reward += trade_amount
        self.day_fees += day_fees
        self.net_reward += trade_amount + day_fees
        self.account_balance += trade_amount + day_fees
        self.lowest_account_balance = min(self.lowest_account_balance, self.account_balance)
        self.highest_account_balance = max(self.highest_account_balance, self.account_balance)
        self.n_wins += 1 if trade_amount > 0 else 0
        self.n_losses += 1 if trade_amount < 0 else 0
        self._curr_win_streak = 0 if trade_amount <= 0 else self._curr_win_streak + 1
        self._curr_loss_streak = 0 if trade_amount >= 0 else self._curr_loss_streak + 1
        self.longest_win_streak = max(self.longest_win_streak, self._curr_win_streak)
        self.longest_loss_streak = max(self.longest_loss_streak, self._curr_loss_streak)

    def __str__(self):
        return f'RESULTS:\nreward = {self.reward}\nday fees = {self.day_fees}\nnet reward = {self.net_reward}' \
               f'\navg pips risked = {self.avg_pips_risked}\nbuys = {self.n_buys}\nsells = {self.n_sells}' \
               f'\nwins = {self.n_wins}\nlosses = {self.n_losses}\nlongest win streak = {self.longest_win_streak}' \
               f'\nlongest loss streak = {self.longest_loss_streak}\nstarting account balance = ' \
               f'{self.starting_account_balance}\nlowest account balance = ' \
               f'{self.lowest_account_balance}\nhighest account balance = {self.highest_account_balance}\n' \
               f'final account balance = {self.account_balance}'
