data = dlmread("./PycharmProjects/Yum/q_table_row_visit.txt");
data2 = data(1:end,2);
sz_data2 = size(data2)  # 516096 as usual

# Number of times a row gets visited
data_no_zeros = data2;
# every 2048th element is a known zero not to be considered
data_no_zeros(2048:2048:end) = [];
size(data_no_zeros)
how_many_zeros = sum(data_no_zeros == 0)  # gives 8394 on old one new one 11588, and then 10022 then 9024 (9M) then 5850 now 0!! (18M scrambled)
[max_visit_val, max_visit_ind] = max(data_no_zeros)
[min_visit_val, min_visit_ind] = min(data_no_zeros)
sorted_data2 = flip(sort(data_no_zeros));
sorted_data2(1:20)
figure(1, 'position', [250, 250, 1250, 1000])
subplot(2, 1, 1)
semilogy(sorted_data2)
grid on
grid minor on
title("Row Visits")


# streak: how many consecutive visits without a change in the action
data_streak = data(1:end,4);
data_streak_no_zeros = data_streak;
data_streak_no_zeros(2048:2048:end) = [];
how_many_zeros_streak = sum(data_streak_no_zeros == 0)
sorted_data_streak_no_zeros = flip(sort(data_streak_no_zeros));
subplot(2, 1, 2)
semilogy(sorted_data_streak_no_zeros)
grid on
grid minor on
title("Streak Duration")
