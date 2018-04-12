import csvimport.*

table = csvimport('train.csv')
headers = table(1,:)
tags = table([2:end],1)
table = table([2:end],[2:end])
