import pandas as pd
import matplotlib.pyplot as plt
from sri_csv import remove_trailing_epochs, calculate_sri, calculate_midpoint, plot_sleep


EPOCHS_PER_DAY = 2880
sleep = remove_trailing_epochs(
    pd.read_csv('test.csv')['sleep'].values
)

plt.figure(figsize=(10,5))
plot_sleep(sleep, fignum=1)

plt.savefig('sri.png')

sri = calculate_sri(sleep, epochs_per_day=EPOCHS_PER_DAY)
midpoint = calculate_midpoint(sleep, epochs_per_day=EPOCHS_PER_DAY)

print('The SRI is %.1f' % sri)
print('The sleep midpoint is epoch %i' % midpoint)