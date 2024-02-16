# Databricks notebook source
import unittest
import os
import pathlib  
import datetime


def find_file(name, starting_path=pathlib.Path().absolute().parent.parent.parent):
    """Finds a file in the folder structure and returns the first matching path to it"""
    for root, dirs, files in os.walk(starting_path):
        if name in files:
            return os.path.join(root, name)
            
            
class TestNotebook(unittest.TestCase):        
    def test_task_1(self):
        assert(all(pd.read_csv(find_file('potus.csv')) == df_potus))
        assert(all(pd.read_csv(find_file('f500.csv')) == df_f500))

    def test_task_2(self):
        expected_state = type(xgboost.XGBClassifier())
        returned_state = type(xgb)
        self.assertEqual(expected_state, returned_state)        
        
    def test_task_3(self): 
        self.assertEqual(multiply_by_three(5), 15, "Your function must have a return statement and must be correct")
        self.assertEqual(multiply_by_three(0), 0, "Your function must have a return statement and must be correct")
        self.assertEqual(multiply_by_three(-4), 0, "Your function must return 0 for any negative integer")

    def test_task_4(self):     
        assert(all(df_2015_new == df_2015.drop("Country", axis=1)))
    
    def test_task_5(self):
        assert(all(row_walmart == df_f500[df_f500['company']=="Walmart"]))
    
    def test_task_6(self):  
        assert(cheeky == 5)
    
    def test_task_7(self): 
        self.assertEqual(bank_account_1.account_name.upper(), 'RBI', 'bank_account_1 account name should be RBI')
        self.assertEqual(bank_account_1.currency.capitalize(), 'Euro', 'bank_account_1 currency should be Euro')
        self.assertEqual(bank_account_1.balance, 25000, 'bank_account_1 balance should be 25000')

        self.assertEqual(bank_account_2.account_name.upper(), 'BT', 'bank_account_2 account name should be BT')
        self.assertEqual(bank_account_2.currency.capitalize(), 'Euro', 'bank_account_2 currency should be Euro')
        self.assertEqual(bank_account_2.balance, 500, 'bank_account_2 balance should be 500')
            
    def test_task_8(self):  
        self.assertEqual(total_balance, 25500, 'The total balance of the bank accounts should be 25500')

    def test_task_9(self): 
        assert(int(diff) == 157)
    
    def test_task_x10(self):
        self.assertIsInstance(five_days_ago, datetime.date)
            
            
unittest.main(argv=[''], verbosity=3, exit=False)
