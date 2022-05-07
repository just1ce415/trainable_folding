import click

VAR = 0


def fun():
    print('dsdf', VAR)


@click.command()
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def main(**kwargs):
    print(kwargs)
    print("ASDF")
    global VAR
    VAR = 234324
    fun()


if __name__ == '__main__':
    main()
