import os
import json
import time
import sqlite3
import streamlit as st

conn = sqlite3.connect('db/database.db', check_same_thread=False)
cursor = conn.cursor()


def take_credentials():
    try:
        key = int(st.text_input('id').split()[0])
        password = st.text_input('password', type='password')

        return key, password
    except:
        pass


def register():
    name = st.text_input('Name')
    password = st.text_input('Password', type='password')

    if st.button('register'):
        cursor.execute("""SELECT MAX(id) from USER""")
        unique_id = cursor.fetchone()[0] + 1
        cursor.execute("""INSERT INTO USER (id, name, password) VALUES (?, ?, ?)""", (unique_id, name, password))
        conn.commit()
        st.success('REGISTERED SUCCESSFULLY !')
        st.write(f"This ID is assigned to you : {unique_id}")
        st.write("You are required to log in with your id. SO DON'T FORGET THIS !")
        update_permits(log=True, reg=False, start=False)
        st.button('click here to login ')


def log_in():
    credentials = take_credentials()
    if st.button('log in'):
        if not exists(credentials):
            st.error('User not registered.')
            update_permits(log=False, reg=True, start=False)
            if st.button('click here to register !'):
                pass
        else:
            with open('session/user.txt', 'w') as file:
                file.write(str(credentials[0]))
            st.success('Logged in successfully!')
            time.sleep(1)
            return True


def sign_out():
    os.remove('session/user.txt')
    update_permits(log=True, reg=False, start=False)


def exists(credentials):
    key, password = credentials
    cursor.execute("""SELECT * FROM USER WHERE id = ? AND password = ?""", (key, password))
    return cursor.fetchone()


def current_user_session():
    with open('session/user.txt', 'r') as file:
        user_session = file.read()
    return user_session


def display():
    records = cursor.execute("""SELECT * FROM USER""")
    for record in records:
        print(record)


def get_permits():
    with open('session/permits.json', 'r') as permit:
        permits = json.load(permit)

    return permits


def update_permits(log, reg, start):
    permits = get_permits()
    permits['log'] = log
    permits['reg'] = reg
    permits['start'] = start

    with open('session/permits.json', 'w') as file:
        json.dump(permits, file)
