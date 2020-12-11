%option noyywrap

%% /* Rules Section */

threadIdx.x ECHO; printf("+1");

%% /* User Code Section */

int main(int argc, char *argv[]) {
  yylex();
}
