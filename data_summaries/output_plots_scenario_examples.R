require(tidyverse)

Lost1 <- read.csv("Lost_slots_results_per_day.csv")
Lost2 <- read.csv("Lost_slots_results_per_day2.csv")
Lost3 <- read.csv("Lost_slots_results_per_day3.csv")
Lost4 <- read.csv("Lost_slots_results_per_day4.csv")

Day1 <- read.csv("audit_day_results_across_runs1.csv")
Day2 <- read.csv("audit_day_results_across_runs2.csv")
Day3 <- read.csv("audit_day_results_across_runs3.csv")
Day4 <- read.csv("audit_day_results_across_runs4.csv")

#take means per day across runs
ts_queue <- out %>% group_by(day) %>% 
  summarise('E(q)_day'=mean(q_length))

Lost1$S <- rep("Baseline", length(Lost1$Day))
Lost2$S <- rep("LoS_reduce_1_day", length(Lost1$Day))
Lost3$S <- rep("Beds_increase_by_6", length(Lost1$Day))
Lost4$S <- rep("Reduce_prop_delayed", length(Lost1$Day))

lost <- rbind(Lost1,Lost2,Lost3,Lost4)

Day1$S <- rep("Baseline", length(Day1$sim_time))
Day2$S <- rep("LoS_reduce_1_day", length(Day1$sim_time))
Day3$S <- rep("Beds_increase_by_6", length(Day1$sim_time))
Day4$S <- rep("Reduce_prop_delayed", length(Day1$sim_time))

day <- rbind(Day1,Day2,Day3,Day4)
day <- day %>% group_by(sim_time, S) %>%
  summarise('DayBedUtil'=mean(bed_utilisation))


#line plot function
plot1 <- function(plotdata){
  plot <- plotdata %>%
    ggplot(aes(x=Day, y=DayLostSlots, color=factor(S)))+
    #scale_x_continuous(breaks=c(4.75,5.0,5.25,5.5))+
    #scale_colour_manual(values=cbPalette)+
    geom_line(lwd=0.9)+
    theme_bw()+
    theme(
      panel.border=element_rect(fill=NA, color="grey50",size = 3.5,linetype="solid"),
      legend.position="bottom", legend.title = element_blank(),
      strip.text.x = element_text(size=15),
      axis.text = element_text(size=15),
      legend.text = element_text(size=15),
      text=element_text(size=15),
      axis.title.x=element_blank(),
      axis.title.y=element_blank())+
    geom_point()
    #geom_errorbar(aes(ymin=CI.lower.limit, ymax=CI.Upper.limit), width=0.04)+
    #facet_grid(metric~., scales="free")
  return(plot)
}

png(filename="BedUtilisation", width =1200, height=700)
plot1(day)
dev.off()

png(filename="LostSlots", width =1200, height=700)
plot1(lost)
dev.off()

head(day)
