import streamlit as st


def login_page():
    st.title("Login Page")
    st.write("Please enter your credentials to log in:")

    # Text input fields for ID and Password
    id_input = st.text_input("ID")
    password_input = st.text_input("Password", type="password")

    # Login and Register buttons
    if st.button("Login"):
        if id_input == "user123" and password_input == "pass123":
            st.success("Login successful!")
        else:
            st.error("Invalid credentials. Please try again.")
            st.button("Register")

    elif st.button("Register"):
        st.session_state["page"] = "register"


def register_page():
    st.title("Registration Page")
    st.write("Please fill in the following details to register:")

    # Text input fields for Username, Password, and Confirm Password
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    confirm_password_input = st.text_input("Confirm Password", type="password")

    # Register button
    if st.button("Register"):
        if password_input == confirm_password_input:
            st.success("Registration successful!")
            st.session_state["page"] = "login"
        else:
            st.error("Passwords do not match. Please try again.")


def main():
    # Initialize session state to keep track of the current page
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if st.session_state["page"] == "login":
        login_page()

    elif st.session_state["page"] == "register":
        register_page()


if __name__ == "__main__":
    main()
