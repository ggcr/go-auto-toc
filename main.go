package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
)

func checkErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	file, err := os.Open("example.md")
	checkErr(err)

	out, err := os.Create("toc.md")
	checkErr(err)

	defer out.Close()

	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Bytes()
		prevFlag := false
		for idx := range line {
			val := line[idx]
			if val == 35 {
				if idx != 0 && idx != 1 {
					out.WriteString("  ")
					fmt.Printf("  ")
					prevFlag = true
				} else if idx == 1 {
					prevFlag = true
				}
			} else if val != 35 && prevFlag {
				s := fmt.Sprintf("- %s\n", line[idx:])
				out.WriteString(strings.Trim(s, "\""))
				fmt.Printf(s)
				break
			} else {
				break
			}
		}
	}
}
