

def pr2(n):
    sumup=0
    for i in [x+1 for x in range(n)]:
        if i%2==0 or i%5==0:
            if i%10 != 0:
                sumup += i
    return sumup

print(pr2(30))

