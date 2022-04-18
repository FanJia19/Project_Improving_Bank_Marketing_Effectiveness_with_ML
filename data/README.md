# Bank Marketing
A retail bank uses its own contact-center to do direct marketing campaigns, mainly through phone
calls (telemarketing). Each campaign is managed in an integrated fashion and the results for all calls
and clients within the campaign are gathered together, in a flat file report concerning only the data
used to do the phone call.

## Scenario
A credit card issuer wants to better predict the likelihood of default for its customers, as well
as identify the key drivers that determine this likelihood. This would inform the issuer’s
decisions on who to give a credit card to and what credit limit to provide. It would also help
the issuer have a better understanding of their current and potential customers, which
would inform their future strategy, including their planning of offering targeted credit
products to their customers.

## Data Description
The credit card issuer has gathered information on 30000 customers. The dataset contains
information on 24 variables, including demographic factors, credit data, history of payment,
and bill statements of credit card customers, as well as information on the outcome: did the
customer default or not? The data to create the model is stored in the file "CreditCardDefault.csv".

**A. Bank client data:**

`age`: age (numeric)

`job`: type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')

`marital`: marital status (categorical: 'divorced', 'married', 'single', 'unknown'; note: 'divorced'
means divorced or widowed)

`education`: (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
'professional.course', 'university.degree', 'unknown')

`default`: has credit in default? (categorical: 'no', 'yes', 'unknown')

`housing`: has housing loan? (categorical: 'no', 'yes', 'unknown')

`loan`: has personal loan? (categorical: 'no', 'yes', 'unknown')

**B. Related with the last contact of the current campaign:**

`contact`: contact communication type (categorical: 'cellular', 'telephone')

`month`: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')

`day_of_week`: last contact day of the week (categorical: 'mon', 'tue', 'wed', 'thu', 'fri')

`duration`: last contact duration, in seconds (numeric). Important note: this attribute highly
affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a
call is performed. Also, after the end of the call y is obviously known. Thus, this input should only
be included for benchmark purposes and should be discarded if the intention is to have a
realistic predictive model.

**C. Other attributes:**


`campaign`: number of contacts performed during this campaign and for this client (numeric,
includes last contact)

`pdays`: number of days that passed by after the client was last contacted from a previous
campaign (numeric; 999 means client was not previously contacted)

`previous`: number of contacts performed before this campaign and for this client (numeric)

`poutcome`: outcome of the previous marketing campaign (categorical: 'failure', 'nonexistent',
'success')

**D. Social and economic context attributes**

`emp.var.rate`: employment variation rate, quarterly indicator (numeric)

`cons.price.idx`: consumer price index, monthly indicator (numeric)

`cons.conf.idx`: consumer confidence index, monthly indicator (numeric)

`euribor3m`: euribor 3 month rate, daily indicator (numeric)

`nr.employed`: number of employees, quarterly indicator (numeric)

**Output variable (desired target):**

`y`: has the client subscribed a term deposit? (binary: 'yes', 'no')
