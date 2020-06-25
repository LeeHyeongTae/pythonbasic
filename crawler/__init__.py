from crawler.controller import Controller

if __name__ == '__main__':
    def print_menu():
        print('0. Exit')
        print('1. bugsMusic')
        return input('Menu\n 크롤링할 사이트를 선택해 주세요.\n')

    app = Controller()
    while 1:
        menu = print_menu()
        if menu == '0':
            print('종료.')
            break
        if menu == '1':
            print('bugsMusic')
            pass