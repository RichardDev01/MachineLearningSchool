# v=6**9
# print(sum([4/(1+i*1/v*i*1/v)for i in range(v)])*1/v)
#
# v = 10**7
# print(sum((4/(1+(i/v)**2)for i in range(v)))*1/v)
#
# v = 9**9
# print(sum((4/(1+(i/v)**2)for i in range(v)))*1/v)

print((6*sum(1/(i*i)for i in range(1,8**8)))/2)