# -*- utf-8 -*-
"""
Created on 14 Aug, 2019.

This module is used as database interface.

@Author: Huang Zewen
"""


import MySQLdb

APT_MYSQL_TOKEN = ('135.252.218.131', 'root', 'apt_AI123', 'APTAnalyzer', 3306)


class APTDB(object):
    def __init__(self, host, user, pwd, db_name, port):
        self._conn = self._cursor = None
        self._connect_db(host, user, pwd, db_name, port)

    def _connect_db(self, host, user, pwd, db_name, port):
        try:
            self._conn = MySQLdb.connect(host=host, port=port,
                                         user=user, passwd=pwd,
                                         db=db_name)

            self._cursor = self._conn.cursor()
        except Exception as err:
            print("Error found while try to connect to database.")
            raise

    def insert(self, sql, *params):
        self._execute(sql, *params)

    def delete(self, sql, *params):
        self._execute(sql, *params)

    def update(self, sql, *params):
        self._execute(sql, *params)

    def _execute(self, sql, *params):
        try:
            self._cursor.execute(sql, params)

            self._cursor.commit()
        except Exception as e:
            print(e)

    def query(self, sql, *params):
        try:
            self._cursor.execute(sql, params)
            return self._cursor.fetchall()
        except Exception as e:
            print("Error while execute query: %s" %e)

    def close(self):
        if self._conn is not None:
            if self._cursor is not None:
                self._cursor.close()

            self._conn.close()

    def get_sh_err_logs_with_label(self):
        sql_str = "SELECT a.Warnings, d.ClassName FROM `NBCaseInfo` AS a, `NBCaseResult` AS b, `AnalyzeRecord` AS c, " \
                  "`Classes` AS d WHERE c.ftid='%s' AND c.aid=b.aid AND a.id=b.case_id AND b.cid=d.cid;"

        # SHANGHAI team's ftid = 1
        params = (3, )
        return self.query(sql_str, *params)


if __name__ == "__main__":
    pass
