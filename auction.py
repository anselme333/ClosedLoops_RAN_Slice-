# VCG auction model for RBS
# Author: Anselme
# Interpreter: python3.8.10
################################################################################
# Needed packages
import random
import time
import cvxpy as cp
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
################################################################################
# Auction using VCG
# Inputs
import pandas as pd

starting_time = time.time()
random.seed(40)
num_tenant = 10
max_iter = num_tenant
# The vector of  bidding values per unit of resource block
bid_tenant = np.random.uniform(low=10, high=20, size=(num_tenant,))
# The vector of RB demands from tenants
tenant_RBs_Req = np.random.uniform(low=6, high=40, size=(num_tenant,)).astype(int)
tenant_RBs_Req_copy = copy.deepcopy(tenant_RBs_Req)
# Total RB of infrastructure provider
# https://www.techplayon.com/nr-resource-block-definition-and-rbs-calculation/
# https://ourtechplanet.com/practical-python-use-cases-session-1-how-to-create-basic-5g-interactive-tools/
# We assume the  channel_bandwidth are the same for the base stations
channel_bandwidth = 100  # Channel bandwidth: 100 MHZ
sub_carrier_spacing = 30  # Sub_carrier spacing (KHZ)
Number_MIMO_layers = 4  # Number of MIMO layers
NumberRB = channel_bandwidth * 1000 / sub_carrier_spacing / 12  # total Resource Block

guard_band = 4  # guard band is an unused part of the radio spectrum between radio bands
Number_PRB = math.floor(NumberRB) - guard_band
InP_max_RBs = Number_PRB
InP_max_RBs_copy = copy.deepcopy(InP_max_RBs)
print("InP_max_RBs", InP_max_RBs)
# print(InP_max_RBs)

# Reserved price per unit of RB
reserved_RB_price = 15
eligible_tenant = []
eligiblle_demands_mvno = []
print("Initial bidding values", bid_tenant)
print("Initial demands", tenant_RBs_Req)

for i in range(0, len(bid_tenant)):
	if bid_tenant[i] >= reserved_RB_price:
		eligible_tenant.append(bid_tenant[i])
		eligiblle_demands_mvno.append(tenant_RBs_Req[i])
print("eligible tenant ",  eligible_tenant)
print("eligible demands ", eligiblle_demands_mvno)

# Just keep the copy of eligible bidding values and demands
eligible_bidding_value = abs(np.array(eligible_tenant))
eligible_bidding_demand = abs(np.array(eligiblle_demands_mvno))
total_RB_capaciy = abs(InP_max_RBs)

###############################################################################
# Initialization
# The welfare of other players than tenant m from the chosen outcome
# when tenant m participates in the auction
v_m = []

# The welfare for other bidders than tenant m from the chosen outcome when tenant
# m is not participating in the auction
v_no_m = []

# Winning  bidding values
w = []

#  Optiaml Payment

P_optimal = np.zeros(len(eligible_tenant))

#  Auction decision variable

x_decision = np.zeros(len(eligible_tenant))
###############################################################################
# Winner determination

for i in range(0, len(eligible_tenant)):
	max_bid_value = max(eligible_tenant)   # Find maximum bidding values
	k = eligible_tenant.index(max_bid_value)
	if eligiblle_demands_mvno[k] <= InP_max_RBs:
		w.append(max_bid_value)
		InP_max_RBs = InP_max_RBs - eligiblle_demands_mvno[k]
		eligible_tenant.remove(eligible_tenant[k])
		eligible_tenant = eligible_tenant
		eligiblle_demands_mvno.remove(eligiblle_demands_mvno[k])
		eligiblle_demands_mvno = eligiblle_demands_mvno
print("winnner value", w)


# just keep a copy of original values
winning_value = abs(np.array(w))
winning_value_f = abs(np.array(w))
L_bidding_values = eligible_bidding_value
L_bidding_demand = eligible_bidding_demand
print("L_bidding_values",L_bidding_values)
print("L_bidding_demand", L_bidding_demand)

new_winning_values = []
###############################################################################
# Price determination
print("winning_value", winning_value)
print("L_bidding_values", L_bidding_values)
for i in range(0, len(winning_value)):
	winner_remove = winning_value[i]
	L_bidding_values = list(L_bidding_values)
	L_bidding_demand = list(L_bidding_demand)
	j = L_bidding_values.index(winner_remove)
	L_bidding_values.remove(L_bidding_values[j])
	L_bidding_demand.remove(L_bidding_demand[j])
	if len(L_bidding_values) != 0:
		new_max_bid_value = max(L_bidding_values)
		# Find maximum bidding values
	new_winning_values.append(new_max_bid_value)
# The chosen outcome when each winning tenant i is not participating and not present
print("New winning values  when each winning tenant i is not participating", new_winning_values)

l = 0
j = len(new_winning_values)
# Choosen outcome, when each winning tenant i is not participating but present
while l <j:
	winnervalue = winning_value[l]
	t=j-1
	other_winning_value = new_winning_values[t]
	others_winners_no_m = list(winning_value[winning_value != winnervalue])
	others_winners_no_m.append(other_winning_value)
	others_winners_than_m = sum(others_winners_no_m)
	v_no_m.append(others_winners_than_m)
	l += 1

	# The welfare of other players than tenant m from the chosen outcome
	# when tenant m is not participating in auction
print("Welfare of other players than tenant m, when m is not there", v_no_m)

print("winning_value_f", winning_value_f)
# The chosen outcome when each winning tenant m is participating in auction

for i in range(0, len(winning_value_f)):
	winner_value = winning_value_f[i]
	others_winners_no_m = winning_value_f[winning_value_f != winner_value]
	others_winner2 = sum(others_winners_no_m)
	v_m.append(others_winner2)
	# The welfare of other players than tenant m from the chosen outcome
	# when tenant m participates in auction
print("Welfare of other players than tenant m, when m is there", v_m)
social_optimal_price = np.array(v_no_m) - np.array(v_m)
print("Social optimal price", social_optimal_price)

for i in range(0, len(w)):
	eligible_bidding_value = list(eligible_bidding_value)
	k = eligible_bidding_value.index(w[i])
	x_decision[k] = 1  # update decision variables based on the winner
	P_optimal[k] = social_optimal_price[i]

# Allocate RB resources to the winners
print("x_decision", x_decision)
print("P_optimal", P_optimal)
RB_allocation_via_auction = eligible_bidding_demand * x_decision
print("RBs allocation VCG", RB_allocation_via_auction)
w_loss = list(set(tenant_RBs_Req).difference(RB_allocation_via_auction))
print("w_loss", w_loss)

# Payments based on demands for RB resources

payments = RB_allocation_via_auction * P_optimal
print("Payments to InP for RB resources", payments)
print("TotalPayment using VCG", sum(payments))
###############################################################################


# using optimization
x = cp.Variable(num_tenant, boolean=True)
objective = cp.Maximize(cp.sum(x @ cp.multiply(bid_tenant, tenant_RBs_Req)))
constraint = [cp.sum(tenant_RBs_Req @ x) <= InP_max_RBs, x @ bid_tenant >= reserved_RB_price, x >= 0, x <= 1]
myprob = cp.Problem(objective, constraint)
result = myprob.solve()
print(x.value)
rb_allocation_optimizer = x.value * tenant_RBs_Req
print("RBs allocation using Mixed-integer Conic Optimizer", rb_allocation_optimizer)

# https://docs.mosek.com/slides/2018/ismp2018/ismp-wiese.pdfhttps://docs.mosek.com/slides/2018/ismp2018/ismp-wiese.pdf
print("TotalPayment using Mixed-integer Conic Optimizer", result)
payment_mosek = bid_tenant * x.value * tenant_RBs_Req


def get_rb_allocation_via_auction():
	return InP_max_RBs, RB_allocation_via_auction


main_list = list(set(tenant_RBs_Req_copy) - set(RB_allocation_via_auction))
for i in range(0, len(main_list)):
	index_losser = np.where(tenant_RBs_Req_copy == main_list[i])
	tenant_RBs_Req_copy[index_losser] = 0
n = 0
print("tenant_RBs_Req_copy",tenant_RBs_Req_copy)
payment_VCG = []
print("P_optimal", P_optimal)
for i in range(len(tenant_RBs_Req_copy)):
	if tenant_RBs_Req_copy[i] > 0 and n < len(P_optimal)+1:
		payments = tenant_RBs_Req_copy[i] * P_optimal[n]
		payment_VCG.append(payments)
		n = n+1
	else:
		payment_VCG.append(0)

print("payment_VCG.", payment_VCG)
payment = bid_tenant * tenant_RBs_Req
payment_ISP = bid_tenant * reserved_RB_price

print("InP_max_RBs ", InP_max_RBs_copy)
RB_allocation_via_auction_percentage = int((sum(RB_allocation_via_auction) * 100)/InP_max_RBs_copy)
rb_allocation_optimizer_percentage = int((sum(rb_allocation_optimizer) * 100)/InP_max_RBs_copy)
print("rb_allocation_VCG_percentage", RB_allocation_via_auction_percentage)
print("rb_allocation_optimizer_percentage", rb_allocation_optimizer_percentage)




plt.plot(tenant_RBs_Req, color='red', linewidth=2.0, marker='o', label='Demands of tenants')
plt.plot(rb_allocation_optimizer, linewidth=2.0, marker = '+', color='yellowgreen',
		 label=f'RBs allocation using MOSEK ({rb_allocation_optimizer_percentage}%)')
plt.plot(tenant_RBs_Req_copy, linewidth=2.0, color='black', marker='D',
		 label=f'RBs allocation using VCG ({RB_allocation_via_auction_percentage}%)')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(color='gray')
plt.xlabel('Tenants', fontsize=18)
plt.ylabel('RBs allocation to tenants', fontsize=18)
plt.ticklabel_format(style='sci')
plt.legend(title='', fontsize=18)
plt.show()

plt.plot(payment, color='red', linewidth=2.0, marker='o', label='Bidding value')
#plt.plot(payment_ISP, linewidth=2.0, marker = '+', color='red', label='Expected payment when all RBs sold')
plt.plot(payment_mosek, linewidth=2.0, marker = '+', color='yellowgreen', label='Payment using MOSEK')
plt.plot(payment_VCG, linewidth=2.0, color='black', marker='D', label='Payment using VCG')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(color='gray')
plt.xlabel('Tenants', fontsize=18)
plt.ylabel('Initial payment', fontsize=18)
plt.ticklabel_format(style='sci')
plt.legend(title='', fontsize=18)
plt.show()


end_time = time.time()
running_time = end_time - starting_time
print("Running Time:", running_time)
# We include InP_max_RBs as the last element for sharing information purpose
RB_allocation_via_auction = np.append(RB_allocation_via_auction, InP_max_RBs_copy)
df = pd.DataFrame()
df["RBS"] = RB_allocation_via_auction
df.to_csv('dataset/RB_allocation_via_auction.csv')