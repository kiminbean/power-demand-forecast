import React from 'react';
import { Text, View } from 'react-native';

export function Hello({ name }: { name: string }) {
  const [greeting] = React.useState('hello');
  return (
    <View accessibilityRole="header">
      <Text>{greeting}</Text>
      <Text>{name}</Text>
    </View>
  );
}
