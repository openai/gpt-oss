# Work in progress

This script uses the Agents SDK in TypeScript and the underlying OpenAI client to verify the shape of the API calls but also whether the API performs tool calling.

1. Update `providers.ts` to create an entry for the API to test. Change `vllm` to the provider name of your choice
2. Run an initial quick test to make sure things work. This will only run one test

```
npm start -- --provider <name> -n 1 -k 1
```

3. Run the full test (runs each test 5 times to test consistency)

```
npm start -- --provider <name> -k 45
```

## Considerations

1. The tests will fail if the API shape does not match the expected behavior
2. Events in the chat API are currently not tested
3. If the schema validation succeeds but the input is wrong the test will still pass for this test. That's because it's likely more of a prompt engineering issue or a validator issue than an API issue as it still nailed the input
