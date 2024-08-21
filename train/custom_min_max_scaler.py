import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from utils.utils import PAD_VAL


# Custom class that uses sklearn's min-max scaler, but ignores padded values
class CustomMinMaxScaler:
    def __init__(self) -> None:
        # Start with an empty scaler
        self._scaler = None

    # Main function - used to scale data
    def scale(self, data: np.array) -> np.array:
        # Replace any padded values with 0 (we don't want the pad values to mess up the min/max calculations)
        data_filtered = np.where(data == PAD_VAL, 0.0, data)

        # If the scaler has not been defined yet, create and fit it on the filtered data
        if self._scaler is None:
            print('FITTING')
            self._scaler = MinMaxScaler()
            self._scaler.fit(data_filtered)

        # Transform the data, reset any padded values
        scaled_data = self._scaler.transform(data_filtered)
        scaled_data[data == PAD_VAL] = PAD_VAL

        return scaled_data

    # Save for later use
    def save(self, file: str) -> None:
        with open(file, 'wb') as f:
            pickle.dump(self, f)
