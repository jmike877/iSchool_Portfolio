### Preprocessing Data
#soccer_df<- data.frame(Soccer_Transfers1)

#soccer_df<- na.omit(soccer_df)
#soccer_df<- soccer_df[soccer_df$transfer_movement == 'in', ]
#soccer_df$transfer_movement[soccer_df$transfer_movement == 'in']<- 'Bought'
#soccer_df$fee_cleaned<- soccer_df$fee_cleaned * 1000000
#soccer_df<- soccer_df[soccer_df$fee_cleaned != 0, ]
#soccer_df$age<- as.numeric(soccer_df$age)
#soccer_df$position[soccer_df$position == 'Left Winger']<- 'Attacker'
#soccer_df$position[soccer_df$position == 'Right Winger']<- 'Attacker'
#soccer_df$position[soccer_df$position == 'Second Striker']<- 'Attacker'
#soccer_df$position[soccer_df$position == 'Centre-Forward']<- 'Attacker'
#soccer_df$position[soccer_df$position == 'Forward']<- 'Attacker'
#soccer_df$position[soccer_df$position == 'attack']<- 'Attacker'
#soccer_df$position[soccer_df$position == 'Defensive Midfield']<- 'Midfielder'
#soccer_df$position[soccer_df$position == 'Right Midfield']<- 'Midfielder'
#soccer_df$position[soccer_df$position == 'Central Midfield']<- 'Midfielder'
#soccer_df$position[soccer_df$position == 'Attacking Midfield']<- 'Midfielder'
#soccer_df$position[soccer_df$position == 'Left Midfield']<- 'Midfielder'
#soccer_df$position[soccer_df$position == 'Midfielder']<- 'Midfielder'
#soccer_df$position[soccer_df$position == 'midfield']<- 'Midfielder'
#soccer_df$position[soccer_df$position == 'Left-Back']<- 'Defender'
#soccer_df$position[soccer_df$position == 'Right-Back']<- 'Defender'
#soccer_df$position[soccer_df$position == 'Centre-Back']<- 'Defender'
#soccer_df$position[soccer_df$position == 'defence']<- 'Defender'
#soccer_df$position[soccer_df$position == 'Winger']<- 'Attacker'
#soccer_df <- subset(soccer_df, select = -c(fee, league_name, 
#                                            club_involved_name, transfer_movement))
#str(soccer_df)
#write.csv(soccer_df,'/Users/jmike877/Desktop/Data Visualization/Final Project/soccer_cleaned.csv',
#          row.names = FALSE)

soccer_df<- read.csv('soccer_cleaned.csv', header = TRUE)

library(ggplot2)
library(dplyr)
library(tidyr)
library(maps)
library(ggpubr)

soccer_average <- soccer_df %>%
  group_by(year, position) %>%
  summarize(avg = mean(fee_cleaned)) %>%
  ungroup()
ggplot(soccer_average %>%
       gather(stat, fee_cleaned, avg), 
       aes(x = year, y = fee_cleaned)) + 
       geom_line(aes(color = position, linetype = stat)) +
       geom_smooth(method = "lm", se = FALSE, col = 'black', size = 1) +
       coord_cartesian(ylim=c(0, 25000000)) + 
       scale_y_continuous(breaks=seq(0, 30000000, 1000000)) +
       theme_classic()

soccer_min <- soccer_df %>%
  group_by(year, position) %>%
  summarize(minimum = min(fee_cleaned)) %>%
  ungroup()
ggplot(soccer_min %>%
       gather(stat, fee_cleaned, minimum), 
       aes(x = year, y = fee_cleaned)) + 
       geom_line(aes(color = position, linetype = stat)) + 
       geom_smooth(method = "lm", se = FALSE, col = 'black', size = 1) +
       coord_cartesian(ylim=c(0, 3000000)) + 
       scale_y_continuous(breaks=seq(0, 5000000, 500000)) + 
       theme_classic()

soccer_max <- soccer_df %>%
  group_by(year, position) %>%
  summarize(maximum = max(fee_cleaned)) %>%
  ungroup()
ggplot(soccer_max %>%
       gather(stat, fee_cleaned, maximum), 
       aes(x = year, y = fee_cleaned)) + 
       geom_line(aes(color = position, linetype = stat)) +
       geom_smooth(method = "lm", se = FALSE, col = 'black', size = 1) +
       coord_cartesian(ylim=c(0, 120000000)) + 
       scale_y_continuous(breaks=seq(0, 150000000, 10000000)) +
       theme_classic()

soccer_df %>%
  count(age, position) %>%
  group_by(factor(age)) %>%
  ggplot() + aes(factor(age), n, fill = position) +
  geom_bar(stat = 'identity') +
  theme_classic()

soccer_df %>%
  count(club_name) %>%
  group_by(factor(club_name)) %>%
  ggplot() + aes(reorder(factor(club_name), -n), n) +
  geom_bar(stat = 'identity') +
  coord_cartesian(ylim=c(0, 150)) + 
  scale_y_continuous(breaks=seq(0, 200, 25)) +
  theme_classic() +
  theme(axis.text.x=element_text(angle = 90, vjust = 1, hjust = 1))

soccer_club_fees <- soccer_df %>%
  group_by(club_name) %>%
  summarize(club_total = sum(fee_cleaned)) %>%
  ungroup()
soccer_club_fees %>%
  group_by(factor(club_name)) %>%
  ggplot() + aes(reorder(factor(club_name), -club_total), club_total) +
  geom_bar(stat = 'identity') +
  coord_cartesian(ylim=c(0, 3000000000)) + 
  scale_y_continuous(breaks=seq(0, 10000000000, 250000000)) +
  theme_classic() +
  theme(axis.text.x=element_text(angle = 90, vjust = 1, hjust = 1))

soccer_window<- soccer_df %>% 
  group_by(transfer_period) %>% 
  count() %>% 
  ungroup() %>% 
  mutate(percentage = n/sum(n)) %>% 
  arrange(desc(transfer_period))
soccer_window$label <- scales::percent(soccer_window$percentage)
ggplot(data = soccer_window) +
  geom_bar(aes(x = "", y = percentage, fill = transfer_period),
           stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  theme_void() +
  geom_text(aes(x = 1, y = cumsum(percentage) - percentage/2, label = label))

soccer_positions<- soccer_df %>% 
  group_by(position) %>% 
  count() %>% 
  ungroup() %>% 
  mutate(percentage = n/sum(n)) %>% 
  arrange(desc(position))
soccer_positions$label <- scales::percent(soccer_positions$percentage)
ggplot(data = soccer_positions) +
  geom_bar(aes(x = "", y = percentage, fill = position),
           stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  theme_void() +
  geom_text(aes(x = 1, y = cumsum(percentage) - percentage/2, label = label))

soccer_club_buys<- soccer_df %>%
  group_by(club_name) %>%
  count() %>%
  ungroup() %>%
  arrange(desc(n))

worldmap = map_data('world')

britain<- ggplot() + geom_polygon(data = worldmap, 
                                  aes(x = long, y = lat, group = group), 
                                  fill = 'gray90', 
                                  color = 'black') + 
                     coord_fixed(ratio = 1.3, 
                     xlim = c(-10, 3), 
                     ylim = c(50, 59)) + 
                     theme_void()
britain

england_coord<- read.csv('england_coord.csv', header = TRUE, sep = ',')
colnames(england_coord)[1] <- 'club_name'
soccer_club_locations<- merge(soccer_club_buys, england_coord, by = 'club_name')
soccer_club_locations2<- merge(soccer_club_fees, england_coord, by = 'club_name')

britain + geom_point(data = soccer_club_locations, 
                     aes(x = long, 
                         y = lat, size = n, color = log(n)), alpha = 1) + 
  scale_size_area(max_size = 7) + 
  scale_color_viridis_c() + 
  theme(legend.position = 'none') + 
  theme(title = element_text(size = 12))

britain + geom_point(data = soccer_club_locations2, 
                     aes(x = long, 
                         y = lat, size = club_total, color = log(club_total)),
                         alpha = 1) + 
  scale_size_area(max_size = 7) + 
  scale_color_viridis_c() + 
  theme(legend.position = 'none') + 
  theme(title = element_text(size = 12))


unique(soccer_df$season)
soccer_2022<- soccer_df[soccer_df$season == '2021/2022', ]
soccer_2021<- soccer_df[soccer_df$season == '2020/2021', ]
soccer_2020<- soccer_df[soccer_df$season == '2019/2020', ]
soccer_2019<- soccer_df[soccer_df$season == '2018/2019', ]
top_buys_2022<- soccer_2022[order(-soccer_2022$fee_cleaned), ][1:5, ]
top_buys_2021<- soccer_2021[order(-soccer_2021$fee_cleaned), ][1:5, ]
top_buys_2020<- soccer_2020[order(-soccer_2020$fee_cleaned), ][1:5, ]
top_buys_2019<- soccer_2019[order(-soccer_2019$fee_cleaned), ][1:5, ]

plot_2022<- ggplot(top_buys_2022, aes(y = reorder(factor(player_name), fee_cleaned),
                                      x = fee_cleaned)) + 
       geom_bar(position="dodge", stat = 'identity') + 
       coord_cartesian(xlim=c(0, 130000000)) + 
       scale_x_continuous(breaks=seq(0, 150000000, 25000000)) +
       theme_classic()
plot_2022
plot_2021<- ggplot(top_buys_2021, aes(y = reorder(factor(player_name), fee_cleaned),
                                                  x = fee_cleaned)) + 
  geom_bar(position="dodge", stat = 'identity') + 
  coord_cartesian(xlim=c(0, 130000000)) + 
  scale_x_continuous(breaks=seq(0, 150000000, 25000000)) +
  theme_classic()
plot_2021
plot_2020<- ggplot(top_buys_2020, aes(y = reorder(factor(player_name), fee_cleaned),
                                      x = fee_cleaned)) + 
  geom_bar(position="dodge", stat = 'identity') + 
  coord_cartesian(xlim=c(0, 130000000)) + 
  scale_x_continuous(breaks=seq(0, 150000000, 25000000)) +
  theme_classic()
plot_2020
plot_2019<- ggplot(top_buys_2019, aes(y = reorder(factor(player_name), fee_cleaned),
                                      x = fee_cleaned)) + 
  geom_bar(position="dodge", stat = 'identity') + 
  coord_cartesian(xlim=c(0, 130000000)) + 
  scale_x_continuous(breaks=seq(0, 150000000, 25000000)) +
  theme_classic()
plot_2019

ggarrange(plot_2022, plot_2021, plot_2020, plot_2019, nrow = 2, ncol = 2)














