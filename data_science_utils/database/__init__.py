import numpy as np  # linear algebra
import pandas as pd
class MySQLDataBaseConnection:
    """
    This Class helps you to use a MySQL DB easily with python. Read and Write Dataframes are supported.
    No Support for connection pooling.

    Constructor: conn_details should be like {"host":"localhost","database":"db","user":"root","password":""}
    connect_each_call determines whether we will use a new connection every call or use the previous connection
    """

    def __init__(self, conn_details, connect_each_call=True):
        from mysql.connector import MySQLConnection, Error
        self.connect_each_call = connect_each_call
        self.conn_details = conn_details
        conn = MySQLConnection(**conn_details)
        if not conn.is_connected():
            raise AssertionError("Could not connect to DB")
        self.conn = conn

    def _create_connection(self):
        from mysql.connector import MySQLConnection, Error
        if not self.connect_each_call:
            if self.conn is None or not self.conn.is_connected():
                raise AssertionError("DB connection closed")
        else:
            if self.conn is not None and self.conn.is_connected():
                self.conn.close()
            conn = MySQLConnection(**self.conn_details)
            if not conn.is_connected():
                raise AssertionError("Could not connect to DB")
            self.conn = conn

    def _handle_connection(self):
        if self.conn is not None and self.connect_each_call:
            self.conn.close()

    def close(self):
        if self.conn is not None and self.conn.is_connected():
            self.conn.close()

    def read_rows_raw_query_get_cursor(self, raw_query):
        self._create_connection()
        conn = self.conn
        cursor = conn.cursor()
        cursor.execute(raw_query)
        return cursor

    def read_rows_raw_query(self, raw_query):
        try:
            self._create_connection()
            conn = self.conn
            cursor = conn.cursor()
            cursor.execute(raw_query)
            rows = cursor.fetchall()
            return rows

        finally:
            cursor.close()
            self._handle_connection()

    def read_rows(self, table_name, cols="*", where_clause=""):
        if not bool(cols):
            raise ValueError("Empty Columns list passed for writing")
        if cols is not None and len(cols) > 0:
            cols = "`,`".join(cols)
        cols = "`"+cols+"`"
        query = "select %s from %s %s" % (cols,table_name, where_clause)
        rows = self.read_rows_raw_query(query)
        return rows

    def read_rows_get_objects(self, table_name, cols="*", where_clause=""):
        rows = self.read_rows(table_name, cols, where_clause)
        if type(cols) is list:
            obj_list = list()
            for row in rows:
                obj = {}
                for i in range(len(cols)):
                    colname = cols[i]
                    obj[colname] = row[i]
                obj_list.append(obj)
            return obj_list
        return rows

    def read_rows_raw_query_get_dataframe(self, query, dataframe_columns=[]):
        data = self.read_rows_raw_query(query)
        df = pd.DataFrame.from_records(data)
        if dataframe_columns is not None and len(dataframe_columns) > 0 and len(dataframe_columns) != len(df.columns):
            raise AssertionError("Columns names for dataframe do not match columns fetched from database")
        if dataframe_columns is not None and len(dataframe_columns) == len(df.columns):
            df.columns = dataframe_columns
        return df

    def read_rows_get_dataframe(self, table_name, cols, where_clause=""):
        if cols is None or type(cols) is not list or len(cols) == 0:
            raise ValueError("Column names not given")
        data = self.read_rows_get_objects(table_name, cols, where_clause)
        return pd.DataFrame.from_records(data)

    def insert_one_row(self, table_name, key_value_pairs):
        if not bool(key_value_pairs):
            raise ValueError("Empty key value pair passed for writing")
        try:
            self._create_connection()
            colnames = "`,`".join([key for key in key_value_pairs.keys()])
            placeholders = ",".join(["%%(%s)s" % key for key in key_value_pairs.keys()])
            colnames = "`" + colnames + "`"
            query = "INSERT INTO %s (%s)" % (table_name, colnames)
            query = query + " VALUES (%s)" % (placeholders)
            cursor = self.conn.cursor()
            cursor.execute(query, key_value_pairs)
            self.conn.commit()
        finally:
            cursor.close()
            self._handle_connection()

    def insert_multiple_rows(self, table_name, cols, values):
        if cols is None or len(cols) == 0:
            raise ValueError("Empty Columns list passed for writing")
        if values is None or len(values) == 0:
            raise ValueError("Empty Values list passed for writing")
        if len(cols) != len(values[0]):
            raise ValueError("Value list and column names length mismatch")
        if type(values[0]) is not tuple:
            values = [tuple(x) for x in values]
        try:
            self._create_connection()
            colnames = "`,`".join([key for key in cols])
            placeholders = ",".join(["%s" for key in cols])
            colnames = "`" + colnames + "`"
            query = "INSERT INTO %s (%s)" % (table_name, colnames)
            query = query + " VALUES (%s)" % (placeholders)
            conn = self.conn
            cursor = conn.cursor()
            cursor.executemany(query, values)
            conn.commit()
        finally:
            cursor.close()
            self._handle_connection()

    def insert_dataframe(self, table_name, df, cols):
        if cols is None or len(cols) == 0:
            raise ValueError("No Columns specified to be used")
        if df is None or len(df) == 0:
            raise ValueError("Empty Dataframe passed")
        self.insert_multiple_rows(table_name, cols, df[cols].values)

    def insert_or_update_rows(self, table_name, cols, values):
        if cols is None or len(cols) == 0:
            raise ValueError("Empty Columns list passed for writing")
        if values is None or len(values) == 0:
            raise ValueError("Empty Values list passed for writing")
        if len(cols) != len(values[0]):
            raise ValueError("Value list and column names length mismatch")
        if type(values[0]) is not tuple:
            values = [tuple(x) for x in values]
        try:
            self._create_connection()
            colnames = "(`" + "`,`".join(cols) + "`)"
            assignments = ",".join(["`{x}` = VALUES(`{x}`)".format(x=x) for x in cols])
            placeholders = ["%s" for x in cols]
            placeholders = "(" + ",".join(placeholders) + ")"

            query = "INSERT INTO %s %s  VALUES %s ON DUPLICATE KEY UPDATE %s" % (
            table_name, colnames, placeholders, assignments)
            conn = self.conn
            cursor = conn.cursor()
            cursor.executemany(query, values)
            conn.commit()
        finally:
            cursor.close()
            self._handle_connection()

    def insert_or_update_dataframe(self, table_name, df, cols):
        if cols is None or len(cols) == 0:
            raise ValueError("No Columns specified to be used")
        if df is None or len(df) == 0:
            raise ValueError("Empty Dataframe passed")
        self.insert_or_update_rows(table_name, cols, df[cols].values)
