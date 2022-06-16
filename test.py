def main():
    print("Hello!")
    str = ""
    lis = []
    while True:
        str = input("Please Enter Food you want to add to your Cart / Or Enter No : ")
        if str == "No":
            break
        else: 
            lis.append(str)
    
    return lis

if __name__ == "__main__":
    temp = []
    temp = main()
    print("Items in the Cart : ")
    for i in temp:
        print(i)
    
    print("Thanks for visiting !, Please visit again")