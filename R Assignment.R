getwd()
flights <- read.csv("flights.csv", header=TRUE, sep = ",") 
head(flights)

# (a)
hist(flights$dep_delay, main="Histogram for flight departure delay", xlab="Departure delay in Minuets", border="blue",col="green",
     xlim=c(0,500),
     breaks=20)
#_____________________________________________________________________________

# (b)
mean(flights$dep_delay)
median(flights$dep_delay)

#____________________________________________________________________________

# (c)
quantile(flights$dep_delay)
sd(flights$dep_delay)
#__________________________________________________________________________
# (d)
quantile(flights$dep_delay, 0.5)
quantile(flights$dep_delay, 0.25)
quantile(flights$dep_delay, 0.95149)

#_____________________________________________________________________________
# (e)
# Scatter Plot
plot(flights$dep_day_period, flights$dep_delay)
# Line Plot
plot(flights$dep_delay,type="o",col=flights$dep_day_period)
# Scatter Plot
ggplot(data = flights, mapping = aes(x = dep_day_period, y = dep_delay)) +
  geom_boxplot()

install.packages("ggplot2")
library("ggplot2")

ggplot(data = flights, mapping = aes(x = dep_day_period, y = dep_delay)) +
  geom_boxplot(alpha = 0) +
  geom_jitter(alpha = 0.3, color = "tomato")



morning <- flights[flights$dep_day_period == "morning", ]
afternoon <- flights[flights$dep_day_period == "afternoon", ]
night <- flights[flights$dep_day_period == "night", ]

# Plot Mornibg
bp_1<-ggplot(data = morning, mapping = aes(x = dep_day_period, y = dep_delay)) +
  geom_boxplot(alpha = 0) +
  geom_jitter(alpha = 0.3, color = "tomato")
bp_1 + ylim(0,200)

# Plot Afternoon
bp_2<-ggplot(data = afternoon, mapping = aes(x = dep_day_period, y = dep_delay)) +
  geom_boxplot(alpha = 0) +
  geom_jitter(alpha = 0.3, color = "tomato")
bp_2 + ylim(0,200)

# Plot night
bp_3<-ggplot(data = night, mapping = aes(x = dep_day_period, y = dep_delay)) +
  geom_boxplot(alpha = 0) +
  geom_jitter(alpha = 0.3, color = "tomato")
bp_3+ ylim(0,200)

summary(morning)
summary(afternoon)
summary(night)
#_______________________________________________________________________________

# (f)(II)
cor.test(flights$dep_delay, flights$arr_delay)
#______________________________________________________________________________

# (g)
mod<-lm(flights$arr_delay ~ flights$dep_delay)
summary(mod)

# END
