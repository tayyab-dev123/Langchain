# def display_name(name, age, city):
#     print(f"{name} {age} {city}")


# display_name(city="SKP", age=26, name="Tayyab")


# Lamda function:

# mul = lambda x, y: x * y

# print(mul(10, 2))


# def apply_operation(x, y, operation):
#     return operation(x, y)


# print(apply_operation(12, 3, lambda a, b: a + b))

# import datetime

# Class


# class Person:

#     def __init__(self, name, age, location) -> None:
#         self.name = name
#         self.age = age
#         self.location = location

#     def talk(self):
#         print(f"{self.name} {self.age} {self.location}")


# Person("Tayyab", 24, "SKP").talk()
# Person("Zainab", 23, "Hvn").talk()


# class Car:
#     num_cars = 0

#     def __init__(self, car, model) -> None:
#         self.car = car
#         self.model = model
#         Car.num_cars += 1


# Car("Honda", 2018)
# Car("Honda", 2018)
# Car("Honda", 2018)
# Car("Honda", 2018)


# print(f"Cars Created {Car.num_cars}")


# class Animal:
#     def __init__(self, name):
#         self.name = name

#     def print_name(self):
#         print(f"Name: {self.name}")


# class Dog(Animal):
#     def __init__(self, name, breed):
#         super().__init__(name)
#         self.name = name
#         self.breed = breed

#     def print_breed(self):
#         print(f"Dog name is {self.name} and breed is {self.breed}")


# Dog("Nido", "German Shepard").print_breed()


# # Duck Typing


# class CreditCardPayment:
#     def process_payment(self, amount):
#         print(f"Processing credit card payment for {amount}")


# class MobilePayment:
#     def process_payment(self, amount):
#         print(f"Processing mobile payment for {amount}")


# def process_payment(payment_method, amount):
#     payment_method.process_payment(amount)


# credit_card_payment = CreditCardPayment()
# mobile_payment = MobilePayment()

# process_payment(
#     credit_card_payment, 100
# )  # Outputs: Processing credit card payment for 100
# process_payment(mobile_payment, 200)  # Outputs: Processing mobile payment for 200


# info = open("User.txt", "r")

# print(info.read())


# import os

# import os
# import pprint

# # Get the list of user's
# env_var = os.environ

# # Print the list of user's
# print("User's Environment variable:")
# pprint.pprint(dict(env_var), width=1)

# 'HOME' environment variable


# import os

# home = os.environ["HOME"]

# print("HOME:", home)

# # 'JAVA_HOME' environment variable
# java_home = os.environ.get("JAVA_HOME")

# # 'JAVA_HOME' environment variable
# print("JAVA_HOME:", java_home)
