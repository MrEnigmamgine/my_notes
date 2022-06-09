'''  get_db_url()
TODO: support multiple keys EG. user= OR username=

Examples:
# Using the env = argument

get_db_url('employees',env='./env.py')

# Using positional arguments

from env import host, username, password
get_db_url('employees',username,password,host)

'''

def get_db_url(database, username='', password='', hostname='', env=''):
    if env != '':
        d = {}
        file = open(env)
        for line in file:
            (key, value) = line.split('=')
            d[key] = value.replace('\n', '').replace("'",'').replace('"','')
        username = d['username']
        hostname = d['hostname'] | d['host']
        password = d['password']
    url = f'mysql+pymysql://{username}:{password}@{hostname}/{database}'
    return url


# Return a float from a string representing a usd amount.
def unstring_usd(str):
    return float(str.replace('$','').replace(',',''))

# Return a USD formatted string from a float.
def as_currency(amount):
    if amount >= 0:
        return '${:,.2f}'.format(amount)
    else:
        return '-${:,.2f}'.format(-amount)