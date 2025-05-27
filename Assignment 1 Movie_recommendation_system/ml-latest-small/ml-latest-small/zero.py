def max_cashback(prod_list):
    n = len(prod_list)
    total_cashback = 0

    while len(prod_list) > 0:
        max_cashback = 0
        max_index = 0

        # Calculate cashback for each item
        for i in range(len(prod_list)):
            left = prod_list[i - 1] if i > 0 else 1
            right = prod_list[i + 1] if i < len(prod_list) - 1 else 1
            cashback = prod_list[i] * left * right

            if cashback > max_cashback:
                max_cashback = cashback
                max_index = i

        # Add the maximum cashback to the total
        total_cashback += max_cashback

        # Remove the item from the list
        prod_list.pop(max_index)

    return total_cashback

# Input reading
n = int(input())
prod_list = list(map(int, input().split()))

# Output the maximum cashback
print(max_cashback(prod_list))
