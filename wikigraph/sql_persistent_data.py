

from collections import MutableMapping
import sqlite3
import pickle

_inc_temp = '''UPDATE articles SET num = num + 1 WHERE name IN ({})'''
_insert_temp = '''INSERT INTO articles VALUES (?,1);'''
_sel_id_temp = '''SELECT rowid FROM articles WHERE name IN ({})'''
_sel_name_temp = '''SELECT name FROM articles WHERE rowid IN ({})'''
_sel_full_id_temp = '''SELECT * FROM articles WHERE rowid IN ({})'''
_sel_full_name_temp = '''SELECT * FROM articles WHERE name IN ({})'''


class Articles(object):
    def __init__(self, db_path, lock=None):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.lock = lock

    def names(self):
        self.cursor.execute('''SELECT name FROM articles''')
        return iter(self.cursor)

    def nums(self):
        self.cursor.execute('''SELECT name FROM articles''')
        return iter(self.cursor)

    def items(self):
        self.cursor.execute('''SELECT * FROM articles''')
        return iter(self.cursor)

    def add(self, *names):
        if self.lock is not None:
            self.lock.acquire()
        for name in names:
            try:
                # print(_insert_temp, name)
                self.cursor.execute(_insert_temp, (name,))
            except sqlite3.IntegrityError:
                pass
        # self.cursor.executemany(_insert_temp, [(n,) for n in names])
        self.conn.commit()
        if self.lock is not None:
            self.lock.release()

    def increment(self, *names):
        if self.lock is not None:
            self.lock.acquire()
        self.cursor.execute(_inc_temp.format(', '.join('?' * len(names))), names)
        self.conn.commit()
        if self.lock is not None:
            self.lock.release()

    def __len__(self):
        self.cursor.execute('''SELECT COUNT(*) FROM articles''')
        return self.cursor.fetchall()[0][0]

    def get_id(self, *names):
        self.cursor.execute(_sel_id_temp.format(', '.join('?' * len(names))), names)
        return self.cursor.fetchall()

    def get_name(self, *IDs):
        self.cursor.execute(_sel_name_temp.format(', '.join('?' * len(IDs))), IDs)
        return self.cursor.fetchall()

    def get_full_from_names(self, *names):
        self.cursor.execute(_sel_full_name_temp.format(', '.join('?' * len(names))), names)
        return self.cursor.fetchall()

    def get_full_from_ids(self, *IDs):
        self.cursor.execute(_sel_full_id_temp.format(', '.join('?' * len(IDs))), IDs)
        return self.cursor.fetchall()

    def __del__(self):
        self.conn.close()

















class PersistentDict(MutableMapping):
    def __init__(self, dbpath, iterable=None, **kwargs):
        self.dbpath = dbpath
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'create table if not exists memo '
                '(key blob primary key not null, value blob not null)'
            )
        if iterable is not None:
            self.update(iterable)
        self.update(kwargs)

    def encode(self, obj):
        return pickle.dumps(obj)

    def decode(self, blob):
        return pickle.loads(blob)

    def get_connection(self):
        return sqlite3.connect(self.dbpath)

    def  __getitem__(self, key):
        key = self.encode(key)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select value from memo where key=?',
                (key,)
            )
            value = cursor.fetchone()
        if value is None:
            raise KeyError(key)
        return self.decode(value[0])

    def __setitem__(self, key, value):
        key = self.encode(key)
        value = self.encode(value)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'insert or replace into memo values (?, ?)',
                (key, value)
            )

    def __delitem__(self, key):
        key = self.encode(key)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select count(*) from memo where key=?',
                (key,)
            )
            if cursor.fetchone()[0] == 0:
                raise KeyError(key)
            cursor.execute(
                'delete from memo where key=?',
                (key,)
            )

    def __iter__(self):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select key from memo'
            )
            records = cursor.fetchall()
        for r in records:
            yield self.decode(r[0])

    def __len__(self):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select count(*) from memo'
            )
            return cursor.fetchone()[0]


class PersistentDict_Simple(MutableMapping):
    def __init__(self, dbpath, iterable=None, **kwargs):
        self.dbpath = dbpath
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'create table if not exists memo '
                '(key blob primary key not null, value blob not null)'
            )
        if iterable is not None:
            self.update(iterable)
        self.update(kwargs)

    def encode(self, obj):
        return pickle.dumps(obj)

    def decode(self, blob):
        return pickle.loads(blob)

    def get_connection(self):
        return sqlite3.connect(self.dbpath)

    def  __getitem__(self, key):
        key = self.encode(key)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select value from memo where key=?',
                (key,)
            )
            value = cursor.fetchone()
        if value is None:
            raise KeyError(key)
        return self.decode(value[0])

    def __setitem__(self, key, value):
        key = self.encode(key)
        value = self.encode(value)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'insert or replace into memo values (?, ?)',
                (key, value)
            )

    def __delitem__(self, key):
        key = self.encode(key)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select count(*) from memo where key=?',
                (key,)
            )
            if cursor.fetchone()[0] == 0:
                raise KeyError(key)
            cursor.execute(
                'delete from memo where key=?',
                (key,)
            )

    def __iter__(self):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select key from memo'
            )
            records = cursor.fetchall()
        for r in records:
            yield self.decode(r[0])

    def __len__(self):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select count(*) from memo'
            )
            return cursor.fetchone()[0]


