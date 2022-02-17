data = dlmread("./PycharmProjects/Yum/score_track_progress_episodes_9.0M_LR_0.5_DIS_0.7.txt");
# data = dlmread("./PycharmProjects/Q_tables_and_backups/score_track_progress_episodes_8.0M_LR_0.15_DIS_0.85_clean.txt");
data2 = data(2:end,:);
wdw_sz = 10
figure(2, 'position', [250, 250, 1250, 1000])
subplot(3, 3, 1)
##plot(data2(:,1), data2(:,3))
plot(movmean(data2(:,4), wdw_sz))
grid on
grid minor on
title("score")
subplot(3, 3, 2)
##plot(data2(:,1), data2(:,4))
plot(movmean(data2(:,5), wdw_sz))
grid on
grid minor on
title("bonus")
subplot(3, 3, 3)
##plot(data2(:,1), data2(:,5))
plot(movmean(data2(:,6), wdw_sz+5))
grid on
grid minor on
title("straight")
subplot(3, 3, 4)
##plot(data2(:,1), data2(:,6))
plot(movmean(data2(:,7), wdw_sz+5))
grid on
grid minor on
title("full")
subplot(3, 3, 5)
##plot(data2(:,1), data2(:,7))
plot(movmean(data2(:,8), wdw_sz+5))
grid on
grid minor on
title("low")
subplot(3, 3, 6)
##plot(data2(:,1), data2(:,8))
plot(movmean(data2(:,9), wdw_sz+5))
grid on
grid minor on
title("high")
subplot(3, 3, 7)
##plot(data2(:,1), data2(:,9))
plot(movmean(data2(:,10), wdw_sz+10))
grid on
grid minor on
title("yum")
subplot(3, 3, 8)
plot(data2(:,1), data2(:,2))
grid on
title("epsilon")
subplot(3, 3, 9)
plot(data2(:,1), data2(:,3))
grid on
title("lr")