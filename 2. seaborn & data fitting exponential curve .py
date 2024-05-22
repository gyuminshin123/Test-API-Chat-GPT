import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["OPENAI_API_KEY"]='sk-proj-Cwt5VDBI2bxarMdQJZNiT3BlbkFJmYc7uqFwdjiIwIXjAcJW'
import openai
import pandas as pd
file = open('Decision questions.csv')
df = pd.read_csv(r'C:/Users/shing/OneDrive - email.ucr.edu/Desktop/API Chat-GPT/Decision questions.csv')

sns.regplot(x = 'keq',
             y = 'binary_answer',
             logistic = True,
             data = df)
plt.ylabel('Probablility of GPT\nchoosing Larger Later')
plt.xlabel('Keq (Larger indicates rational to choose LL)')
plt.show()




###using scipy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy import optimize
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker


x_array = df['keq'].to_numpy()
y_array_exp = df['binary_answer'].to_numpy()

# define a fitting function called exponentail which takes
# in the x-data (x) and returns an exponential curve with equation
# a*exp(x*k) which best fits the data


#def exponential(x, a, k, b):
#    return a*np.exp(x*k) + b
def exponential(x, a, k, b):
    return a*np.exp(x*k) + b

# using the scipy library to fit the x- and y-axis data 
# p0 is where you give the function guesses for the fitting parameters
# this function returns:
#   popt_exponential: this contains the fitting parameters
#   pcov_exponential: estimated covariance of the fitting paramters

popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_array, y_array_exp, p0=[1,-0.5, 1])


# we then can find the error of the fitting parameters
# from the pcov_linear array
perr_exponential = np.sqrt(np.diag(pcov_exponential))

# this cell prints the fitting parameters with their errors
print("pre-exponential factor = %0.2f (+/-) %0.2f" % (popt_exponential[0], perr_exponential[0]))
print("rate constant = %0.2f (+/-) %0.2f" % (popt_exponential[1], perr_exponential[1]))


#ploting the graph with best fit curve AND LEGEND
fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])

ax1.plot(x_array, y_array_exp, "ro")
ax1.plot(x_array, exponential(x_array, *popt_exponential), 'k--', \
         label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))

ax1.set_xlim(0.0001,0.2)
ax1.set_ylim(-1,2)

ax1.set_xlabel("keq",family="serif",  fontsize=12)
ax1.set_ylabel("binary_answer",family="serif",  fontsize=12)

ax1.legend(loc='best')

ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

fig.tight_layout()
fig.savefig("keqBinary_Answer.png", format="png",dpi=1000)


