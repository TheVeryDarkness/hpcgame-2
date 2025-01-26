void f(const void **p) {}
int main() {
    void *p = 0;
    void **q = &p;
    f(q);
}